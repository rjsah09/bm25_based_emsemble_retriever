import pickle
import uuid
import heapq
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
import ahocorasick

# 사용자가 가진 파서 모듈
from pptx_parser import PPTXParser


class BM25Persistent:
    def __init__(self, pickle_path: str, default_k: int = 50):
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

        # 초기 인덱스 구축
        self._build_retriever(k=self.default_k)

    # READ
    def retrieve(
        self,
        query: str,
        k: Optional[int] = 50,
        apply_coordination: bool = False,
    ) -> List[Dict]:
        if not query or not self.collection:
            print("query 또는 collection이 없습니다.")
            return []

        if self._bm25 is None or not self._doc_ids:
            print("BM25 인덱스가 비어있습니다.")
            return []

        k = int(k or self.default_k)

        tokenized_query = self._tokenize_text(query).upper().split()
        if not tokenized_query:
            return []

        # 쿼리 단어 집합 생성 (중복 제거)
        query_words_set = set(tokenized_query)
        query_word_count = len(query_words_set)

        scores = self._bm25.get_scores(tokenized_query)
        if scores.size == 0:
            return []

        # Coordination 점수 계산 및 BM25 점수와 결합
        # ahocorasick Automaton 생성 (키워드들에 대한 Automaton)
        automaton = None
        if apply_coordination and query_word_count > 0:
            automaton = ahocorasick.Automaton()
            for keyword in query_words_set:
                automaton.add_word(keyword, keyword)
            automaton.make_automaton()

        combined_scores = []
        for idx in range(len(scores)):
            bm25_score = float(scores[idx])

            if apply_coordination and query_word_count > 0:
                # 문서의 원본 내용 가져오기
                doc_id = self._doc_ids[idx]
                doc = self._doc_by_id.get(doc_id)
                if doc and automaton:
                    original_content = doc.get("original_content", "")
                    if not isinstance(original_content, str):
                        original_content = str(original_content)

                    # ahocorasick을 사용하여 문서에서 매칭되는 키워드 찾기
                    matched_keywords = set()
                    for end_index, keyword in automaton.iter(original_content):
                        matched_keywords.add(keyword)

                    # Coordination 계산: 매칭된 키워드 수 / 전체 쿼리 키워드 수
                    coordination_score = len(matched_keywords) / query_word_count

                    COORDINATION_SCALE = 1000.0  # coordination에 곱할 스케일
                    combined_score = (
                        coordination_score * COORDINATION_SCALE + bm25_score
                    )
                else:
                    coordination_score = 0.0
                    combined_score = bm25_score
            else:
                combined_score = bm25_score

            combined_scores.append(combined_score)

        top_idx = heapq.nlargest(
            k, range(len(combined_scores)), key=combined_scores.__getitem__
        )

        sel_scores = [combined_scores[i] for i in top_idx]

        # 항상 0~100 범위로 정규화 (apply_coordination 여부와 관계없이)
        if len(sel_scores) > 0:
            mn, mx = min(sel_scores), max(sel_scores)
            if mx > mn:
                # Min-Max 정규화를 0~100 범위로 스케일링
                sel_scores = [((s - mn) / (mx - mn)) * 100.0 for s in sel_scores]
            else:
                # 모든 점수가 같으면 100점으로 설정
                sel_scores = [100.0 for _ in sel_scores]

        out: List[Dict] = []
        for rank, (idx, s) in enumerate(zip(top_idx, sel_scores), start=1):
            doc_id = self._doc_ids[idx]
            d = self._doc_by_id.get(doc_id)
            if not d:
                continue

            item = dict(d)
            item["similarity"] = float(s)
            item["rank"] = rank

            # coordination 점수도 함께 저장 (디버깅/분석용)
            if apply_coordination and query_word_count > 0 and automaton:
                original_content = d.get("original_content", "")
                if not isinstance(original_content, str):
                    original_content = str(original_content)

                # ahocorasick을 사용하여 문서에서 매칭되는 키워드 찾기
                matched_keywords = set()
                for end_index, keyword in automaton.iter(original_content):
                    matched_keywords.add(keyword)

                coordination_score = len(matched_keywords) / query_word_count
                item["coordination"] = coordination_score
                item["matched_query_words"] = len(matched_keywords)
                item["total_query_words"] = query_word_count
                # 디버깅용으로 원래 BM25 점수도 보고 싶으면:
                item["bm25_score"] = float(scores[idx])

            out.append(item)

        return out

    # INSERT
    def save(self, path: str) -> List[Dict]:
        """
        업서트 규칙:
        - 같은 file_name 의 기존 데이터는 메모리/피클에서 모두 제거 후 새로 insert.
        - 피클 저장 1회, 인덱스 재구축 1회로 병목 방지.
        """
        try:
            parser = PPTXParser(path)
            slides_data = parser.parse_slides()
            if not slides_data:
                print(f"'{path}'에서 슬라이드가 비어있습니다.")
                return []

            # 저장 대상 file_name 추출 (보통 파일 내 동일)
            target_file_names = {s.get("file_name", "") for s in slides_data}
            if "" in target_file_names:
                # 안전 장치: 비어있다면 실제 파일명으로 대체
                actual_name = Path(path).name
                target_file_names.discard("")
                target_file_names.add(actual_name)

            # 기존 동일 file_name 데이터 일괄 제거 (메모리)
            deleted = self._drop_by_file_names(target_file_names)

            # 새 문서들 구성
            new_documents: List[Dict] = []
            for slide in slides_data:
                original_text = slide.get("text_content", "") or ""
                tokenized_content = self._tokenize_text(original_text)
                doc = {
                    "id": str(uuid.uuid4()),
                    "file_name": slide.get("file_name", "")
                    or next(iter(target_file_names)),
                    "file_path": slide.get("file_path", "") or path,
                    "slide_index": slide.get("index", -1),
                    "original_content": original_text,
                    "tokenized_content": tokenized_content,
                }
                new_documents.append(doc)

            # 메모리에 반영
            self.collection.extend(new_documents)

            # 피클 1회 저장
            with open(self.pickle_path, "wb") as f:
                pickle.dump(self.collection, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 인덱스 1회 재구축
            self._build_retriever()

            print(
                f"업서트 완료: 제거 {deleted}개 → 추가 {len(new_documents)}개 (파일: {', '.join(sorted(target_file_names))})"
            )
            return new_documents

        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")
            return []

    def save_from_path(self, dir_path: str) -> List[Dict]:
        """
        폴더 배치 업서트:
        - 모든 PPTX 먼저 파싱하여 (file_name -> 새 슬라이드들) 수집
        - 해당 file_name 들을 한 번에 삭제
        - 새 슬라이드들 한 번에 추가
        - 피클 저장 1회 + 인덱스 재구축 1회
        => 대량 처리 시 병목 제거
        """
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

            # 1) 먼저 모두 파싱해서 메모리에 모으기 (추가분)
            staged_new_docs: List[Dict] = []
            staged_file_names: set[str] = set()
            success = 0

            for pptx_file in pptx_files:
                print(f"파싱 중: {pptx_file.name}")
                try:
                    parser = PPTXParser(str(pptx_file))
                    slides_data = parser.parse_slides()
                    if not slides_data:
                        print(f"  - 슬라이드 없음: {pptx_file.name}")
                        continue

                    file_names_here = {s.get("file_name", "") for s in slides_data}
                    if "" in file_names_here:
                        file_names_here.discard("")
                        file_names_here.add(pptx_file.name)

                    representative_name = next(iter(file_names_here))
                    staged_file_names.update(file_names_here)

                    new_docs_this_file: List[Dict] = []
                    for slide in slides_data:
                        original_text = slide.get("text_content", "") or ""
                        tokenized_content = self._tokenize_text(original_text)
                        doc = {
                            "id": str(uuid.uuid4()),
                            "file_name": slide.get("file_name", "")
                            or representative_name,
                            "file_path": slide.get("file_path", "") or str(pptx_file),
                            "slide_index": slide.get("index", -1),
                            "original_content": original_text,
                            "tokenized_content": tokenized_content,
                        }
                        new_docs_this_file.append(doc)

                    staged_new_docs.extend(new_docs_this_file)
                    success += 1
                    print(
                        f"  - {pptx_file.name} 파싱 완료 (+{len(new_docs_this_file)} 슬라이드)"
                    )

                except Exception as e:
                    print(f"[WARN] {pptx_file.name} 파싱 실패: {e}")

            if not staged_new_docs:
                print("추가할 슬라이드가 없습니다.")
                return []

            deleted = self._drop_by_file_names(staged_file_names)

            self.collection.extend(staged_new_docs)

            with open(self.pickle_path, "wb") as f:
                pickle.dump(self.collection, f, protocol=pickle.HIGHEST_PROTOCOL)

            self._build_retriever()

            print(f"\n전체 처리 완료: {success}/{len(pptx_files)}개 파일 파싱 성공")
            print(
                f"기존 삭제: {deleted}개 / 신규 추가: {len(staged_new_docs)}개 / 대상 파일: {len(staged_file_names)}개"
            )
            return staged_new_docs

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
            print(f"모든 데이터가 삭제되었습니다. ({deleted}개 데이터)")
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
            return

        corpus_tokens: List[List[str]] = []
        self._doc_ids = []
        self._doc_by_id = {}

        for d in self.collection:
            tokenized = d.get("tokenized_content", "")
            if not tokenized:
                tokenized = self._tokenize_text(d.get("original_content", "") or "")
                d["tokenized_content"] = tokenized

            # tokenized가 리스트인 경우 문자열로 변환
            if isinstance(tokenized, list):
                tokenized = " ".join(tokenized)
            toks = tokenized.upper().split()
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

    def _drop_by_file_names(self, target_file_names: Iterable[str]) -> int:
        """
        주어진 file_name 집합에 해당하는 문서를 메모리에서 일괄 제거.
        (피클/인덱스 반영은 호출부에서 한 번에 수행)
        """
        target = {fn for fn in target_file_names if fn}
        if not target:
            return 0

        before = len(self.collection)
        self.collection = [
            d for d in self.collection if d.get("file_name") not in target
        ]
        deleted = before - len(self.collection)
        return deleted
