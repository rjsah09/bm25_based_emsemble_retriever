import pickle
import uuid
import heapq
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document  # 선택 사용 (메타데이터 호환용)
from langchain_community.retrievers import BM25Retriever  # 선택 유지
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

# 사용자가 가진 파서 모듈
from pptx_parser import PPTXParser


class BM25Persistent:
    def __init__(
        self, pickle_path: str, use_langchain_bm25: bool = False, default_k: int = 50
    ):
        self.pickle_path = pickle_path
        self.default_k = int(default_k)

        # 토크나이저
        self.kiwi = Kiwi()
        self.stopwords = Stopwords()

        # pickle 로드 또는 초기화
        if not Path(pickle_path).exists():
            print(f"Pickle 파일이 존재하지 않습니다. 새로 생성합니다: {pickle_path}")
            self.collection: List[Dict] = []
            with open(pickle_path, "wb") as f:
                pickle.dump(self.collection, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(pickle_path, "rb") as f:
                self.collection: List[Dict] = pickle.load(f)

        # 인덱스 구조
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []  # 인덱스 -> 문서 id
        self._doc_by_id: Dict[str, Dict] = {}  # id -> 문서 사전

        # (옵션) LangChain BM25 유지
        self._retriever = None
        self._use_langchain_bm25 = bool(use_langchain_bm25)

        # 초기 인덱스 구축
        self._build_retriever(k=self.default_k)

    # READ
    def retrieve(
        self, query: str, k: Optional[int] = 50, normalize_scores: bool = False
    ) -> List[Dict]:
        if not query or not self.collection:
            print("query 또는 collection이 없습니다.")
            return []

        if self._bm25 is None or not self._doc_ids:
            print("BM25 인덱스가 비어있습니다.")
            return []

        k = int(k or self.default_k)

        # 1) 토크나이즈 + upper()
        tokenized_query = self._tokenize_text(query).upper().split()
        if not tokenized_query:
            return []

        # 2) 점수 계산
        scores = self._bm25.get_scores(tokenized_query)
        if scores.size == 0:
            return []

        # 3) 상위 k 추출
        top_idx = heapq.nlargest(k, range(len(scores)), key=scores.__getitem__)

        # 4) 정규화 옵션
        sel_scores = [float(scores[i]) for i in top_idx]
        if normalize_scores and len(sel_scores) > 0:
            mn, mx = min(sel_scores), max(sel_scores)
            if mx > mn:
                sel_scores = [(s - mn) / (mx - mn) for s in sel_scores]
            else:
                sel_scores = [1.0 for _ in sel_scores]

        # 5) 결과 구성
        out: List[Dict] = []
        for rank, (idx, s) in enumerate(zip(top_idx, sel_scores), start=1):
            doc_id = self._doc_ids[idx]
            d = self._doc_by_id.get(doc_id)
            if not d:
                continue

            # 결과 복사 + 점수/순위 부여
            item = dict(d)
            item["similarity"] = float(s)
            item["rank"] = rank
            out.append(item)

        return out

    # INSERT
    def save(self, path: str) -> List[Dict]:
        try:
            parser = PPTXParser(path)
            slides_data = parser.parse_slides()
            new_documents: List[Dict] = []

            for slide in slides_data:
                original_text = slide.get("text_content", "") or ""
                tokenized_content = self._tokenize_text(
                    original_text
                )  # 공백구분 문자열
                doc = {
                    "id": str(uuid.uuid4()),
                    "file_name": slide.get("file_name", ""),
                    "file_path": slide.get("file_path", ""),
                    "slide_index": slide.get("index", -1),
                    "original_content": original_text,  # 원문
                    "tokenized_content": tokenized_content,  # 토크나이즈 결과 (lower는 사용 시점에서)
                    # 호환 필드(이전 코드의 upper_content를 참조하는 로직을 대비)
                    "upper_content": original_text.upper() if original_text else "",
                }
                new_documents.append(doc)

            # extend & persist
            self.collection.extend(new_documents)
            with open(self.pickle_path, "wb") as f:
                pickle.dump(self.collection, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 인덱스 재구축
            self._build_retriever(k=self.default_k)

            print(
                f"'{path}' 파일에서 {len(new_documents)}개 슬라이드를 파싱하여 저장 완료."
            )
            return new_documents

        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")
            return []

    def save_from_path(self, dir_path: str) -> List[Dict]:
        try:
            p = Path(dir_path)
            if not p.exists():
                print(f"디렉토리가 존재하지 않습니다: {dir_path}")
                return []
            if not p.is_dir():
                print(f"경로가 디렉토리가 아닙니다: {dir_path}")
                return []

            pptx_files = list(p.glob("*.pptx"))
            if not pptx_files:
                print(f"디렉토리 '{dir_path}'에서 PPTX 파일을 찾을 수 없습니다.")
                return []

            print(f"'{dir_path}'에서 {len(pptx_files)}개의 PPTX 파일을 발견했습니다.")

            all_saved: List[Dict] = []
            success = 0
            for pptx_file in pptx_files:
                print(f"처리 중: {pptx_file.name}")
                try:
                    saved = self.save(str(pptx_file))
                    if saved:
                        all_saved.extend(saved)
                        success += 1
                        print(f"{pptx_file.name} 처리 완료 ({len(saved)}개 슬라이드)")
                    else:
                        print(f"{pptx_file.name} 처리 실패")
                except Exception as e:
                    print(f"{pptx_file.name} 처리 중 오류: {e}")

            print(f"\n전체 처리 완료: {success}/{len(pptx_files)}개 파일 성공")
            print(f"총 {len(all_saved)}개 슬라이드가 저장되었습니다.")
            return all_saved

        except Exception as e:
            print(f"디렉토리 처리 중 오류 발생: {e}")
            return []

    def save_from_external_path(self, path: str):
        """추후 구현 예정"""
        print("save_from_external_path는 아직 구현되지 않았습니다.")

    # DELETE
    def remove(self, file_name: str) -> int:
        try:
            before = len(self.collection)
            self.collection = [
                d for d in self.collection if d.get("file_name") != file_name
            ]
            deleted = before - len(self.collection)

            if deleted > 0:
                with open(self.pickle_path, "wb") as f:
                    pickle.dump(self.collection, f, protocol=pickle.HIGHEST_PROTOCOL)
                self._build_retriever(k=self.default_k)
                print(f"'{file_name}' 파일의 {deleted}개 슬라이드가 삭제되었습니다.")
            else:
                print(f"'{file_name}' 파일을 찾을 수 없습니다.")

            return deleted

        except Exception as e:
            print(f"데이터 삭제 중 오류 발생: {e}")
            return 0

    def remove_all(self) -> int:
        try:
            deleted = len(self.collection)
            self.collection = []
            with open(self.pickle_path, "wb") as f:
                pickle.dump(self.collection, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._build_retriever(k=self.default_k)
            print(f"모든 데이터가 삭제되었습니다. ({deleted}개 슬라이드)")
            return deleted
        except Exception as e:
            print(f"전체 데이터 삭제 중 오류 발생: {e}")
            return 0

    # INTERNAL FUNCTIONS #
    def _build_retriever(self, k: int = 50) -> None:
        if not self.collection:
            self._bm25 = None
            self._doc_ids = []
            self._doc_by_id = {}
            self._retriever = None
            return

        corpus_tokens: List[List[str]] = []
        documents_for_langchain: List[Document] = []
        self._doc_ids = []
        self._doc_by_id = {}

        for d in self.collection:
            tokenized = d.get("tokenized_content", "")
            if not tokenized:
                tokenized = self._tokenize_text(d.get("original_content", "") or "")
                d["tokenized_content"] = tokenized

            toks = tokenized.lower().split()
            corpus_tokens.append(toks)

            self._doc_ids.append(d["id"])
            self._doc_by_id[d["id"]] = d

        # rank_bm25 인덱스 1회 구축
        self._bm25 = BM25Okapi(corpus_tokens)

    def _tokenize_text(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        tokens = self.kiwi.tokenize(text.strip(), stopwords=self.stopwords)

        KEEP = {
            "NNG",
            "NNP",
            "NNB",
            "NR",
            "NP",
            "VV",
            "VA",
            "VX",
            "VCP",
            "VCN",
            "MM",
            "MAG",
            "MAJ",
            "SL",
            "SN",
        }
        meaningful = [t.form for t in tokens if t.tag in KEEP]
        return " ".join(meaningful)
