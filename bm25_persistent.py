import pickle
import uuid
import math
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pptx_parser import PPTXParser
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords


class BM25Persistent:
    def __init__(self, pickle_path: str):
        self.pickle_path = pickle_path

        # kiwipiepy 토크나이저 초기화
        self.kiwi = Kiwi()
        self.stopwords = Stopwords()

        # pickle 파일이 존재하지 않으면 빈 collection으로 생성
        if not Path(pickle_path).exists():
            print(f"Pickle 파일이 존재하지 않습니다. 새로 생성합니다: {pickle_path}")
            self.collection: list[dict] = []
            with open(pickle_path, "wb") as f:
                pickle.dump(self.collection, f)
        else:
            # 기존 pickle 파일 로드
            with open(pickle_path, "rb") as f:
                self.collection: list[dict] = pickle.load(f)

        self._retriever = None
        self._build_retriever(k=50)

    # READ
    def retrieve(self, query):
        if not query or not self.collection:
            print("query 또는 collection이 없습니다.")
            return []

        # 쿼리도 토크나이징 적용
        tokenized_query = self._tokenize_text(query)
        chunks = self._retriever.get_relevant_documents(tokenized_query.upper())
        out = []

        for d in chunks:
            raw = dict(d.metadata) if isinstance(d.metadata, dict) else {}
            if "upper_content" not in raw:
                raw["upper_content"] = d.page_content or ""

            # BM25 점수 계산 (간단한 텍스트 유사도 기반)
            raw["similarity"] = self._calculate_bm25_similarity(
                tokenized_query, raw.get("tokenized_content", "")
            )
            out.append(raw)
        return out

    # CREATE
    def save(self, path: str):
        """PPTX 파일을 파싱하여 pickle로 저장하고 메모리에 적재"""
        try:
            # PPTXParser를 사용하여 슬라이드 데이터 파싱
            parser = PPTXParser(path)
            slides_data = parser.parse_slides()

            # 각 슬라이드 데이터를 BM25 형식으로 변환
            new_documents = []
            for slide in slides_data:
                # 원본 텍스트를 토크나이징
                tokenized_content = self._tokenize_text(slide["text_content"])

                doc = {
                    "id": str(uuid.uuid4()),
                    "file_name": slide["file_name"],
                    "file_path": slide["file_path"],
                    "slide_index": slide["index"],
                    "upper_content": tokenized_content.upper(),
                    "original_content": slide["text_content"],
                    "tokenized_content": tokenized_content,
                }
                new_documents.append(doc)

            # 기존 collection에 새 데이터 추가
            self.collection.extend(new_documents)

            # pickle 파일로 저장
            # self.remove(new_documents[0]["file_name"])
            with open(self.pickle_path, "wb") as f:
                pickle.dump(self.collection, f)

            # BM25 retriever 재구성
            self._build_retriever()

            print(
                f"'{path}' 파일에서 {len(new_documents)}개 슬라이드를 파싱하여 저장 완료."
            )
            return new_documents

        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")
            return []

    def save_from_path(self, dir_path: str):
        """디렉토리 내 모든 PPTX 파일을 찾아서 각각 save() 함수 호출"""
        try:
            dir_path = Path(dir_path)
            if not dir_path.exists():
                print(f"디렉토리가 존재하지 않습니다: {dir_path}")
                return []

            if not dir_path.is_dir():
                print(f"경로가 디렉토리가 아닙니다: {dir_path}")
                return []

            # 디렉토리 내 모든 PPTX 파일 찾기
            pptx_files = list(dir_path.glob("*.pptx"))

            if not pptx_files:
                print(f"디렉토리 '{dir_path}'에서 PPTX 파일을 찾을 수 없습니다.")
                return []

            print(f"'{dir_path}'에서 {len(pptx_files)}개의 PPTX 파일을 발견했습니다.")

            all_saved_documents = []
            successful_files = 0

            for pptx_file in pptx_files:
                print(f"처리 중: {pptx_file.name}")
                try:
                    saved_docs = self.save(str(pptx_file))
                    if saved_docs:
                        all_saved_documents.extend(saved_docs)
                        successful_files += 1
                        print(
                            f"{pptx_file.name} 처리 완료 ({len(saved_docs)}개 슬라이드)"
                        )
                    else:
                        print(f"{pptx_file.name} 처리 실패")
                except Exception as e:
                    print(f"{pptx_file.name} 처리 중 오류: {e}")

            print(f"\n전체 처리 완료: {successful_files}/{len(pptx_files)}개 파일 성공")
            print(f"총 {len(all_saved_documents)}개 슬라이드가 저장되었습니다.")

            return all_saved_documents

        except Exception as e:
            print(f"디렉토리 처리 중 오류 발생: {e}")
            return []

    def save_from_external_path(self, path: str):
        pass

    # DELETE
    def remove(self, file_name: str):
        """메모리와 pickle 파일에서 특정 file_name과 일치하는 모든 데이터를 삭제"""
        try:
            # 삭제할 데이터 개수 확인
            initial_count = len(self.collection)

            # file_name과 일치하는 데이터 필터링 (제거)
            self.collection = [
                doc for doc in self.collection if doc.get("file_name") != file_name
            ]

            # 삭제된 데이터 개수 계산
            deleted_count = initial_count - len(self.collection)

            if deleted_count > 0:
                # pickle 파일에 업데이트된 collection 저장
                with open(self.pickle_path, "wb") as f:
                    pickle.dump(self.collection, f)

                # BM25 retriever 재구성
                self._build_retriever()

                print(
                    f"'{file_name}' 파일의 {deleted_count}개 슬라이드가 삭제되었습니다."
                )
                return deleted_count
            else:
                print(f"'{file_name}' 파일을 찾을 수 없습니다.")
                return 0

        except Exception as e:
            print(f"데이터 삭제 중 오류 발생: {e}")
            return 0

    def remove_all(self):
        """메모리와 pickle 파일에서 모든 데이터를 삭제"""
        try:
            # 삭제할 데이터 개수 확인
            deleted_count = len(self.collection)

            # collection을 빈 리스트로 초기화
            self.collection = []

            # pickle 파일에 빈 collection 저장
            with open(self.pickle_path, "wb") as f:
                pickle.dump(self.collection, f)

            # BM25 retriever 재구성 (빈 상태로)
            self._build_retriever()

            print(f"모든 데이터가 삭제되었습니다. ({deleted_count}개 슬라이드)")
            return deleted_count

        except Exception as e:
            print(f"전체 데이터 삭제 중 오류 발생: {e}")
            return 0

    def _build_retriever(self, k: int = 50) -> None:
        """collection으로 LangChain BM25Retriever 구성."""
        if not self.collection:
            self._retriever = None
            return
        documents: list[Document] = []
        for d in self.collection:
            # 토크나이징된 내용 사용 (기존 데이터 호환성을 위해 fallback)
            page_content = (
                d.get("upper_content") or d.get("tokenized_content", "").upper() or ""
            )
            documents.append(Document(page_content=page_content, metadata=d))
        self._retriever = BM25Retriever.from_documents(documents, k=50)

    def _tokenize_text(self, text: str) -> str:
        """kiwipiepy를 사용하여 한국어 텍스트를 토크나이징"""
        if not text or not text.strip():
            return ""

        # kiwipiepy로 형태소 분석
        tokens = self.kiwi.tokenize(text.strip(), stopwords=self.stopwords)

        # 명사, 동사, 형용사, 부사만 추출 (불용어 제거)
        meaningful_tokens = []
        for token in tokens:
            pos = token.tag
            word = token.form

            # 의미 있는 품사만 추출
            if pos in [
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
            ]:
                meaningful_tokens.append(word)

        return " ".join(meaningful_tokens)

    def _calculate_bm25_similarity(self, query: str, document: str) -> float:
        """쿼리와 문서 간의 BM25 기반 유사도 점수 계산"""
        if not query or not document:
            return 0.0

        query_tokens = set(query.lower().split())
        doc_tokens = document.lower().split()

        if not query_tokens or not doc_tokens:
            return 0.0

        # 전체 문서 집합에서 IDF 계산을 위한 준비
        total_docs = len(self.collection)
        if total_docs == 0:
            return 0.0

        # 각 문서의 토큰화된 내용을 수집
        all_docs_tokens = []
        for doc in self.collection:
            tokenized_content = doc.get("tokenized_content", "")
            if tokenized_content:
                all_docs_tokens.append(tokenized_content.lower().split())

        # 현재 문서의 토큰 빈도 계산
        doc_token_count = {}
        for token in doc_tokens:
            doc_token_count[token] = doc_token_count.get(token, 0) + 1

        # BM25 스코어 계산
        k1 = 1.2
        b = 0.75
        doc_length = len(doc_tokens)

        # 전체 문서의 평균 길이 계산
        if all_docs_tokens:
            avg_doc_length = sum(len(tokens) for tokens in all_docs_tokens) / len(
                all_docs_tokens
            )
        else:
            avg_doc_length = doc_length

        score = 0.0
        for term in query_tokens:
            if term in doc_token_count:
                tf = doc_token_count[term]

                # TF
                docs_containing_term = 0
                for tokens in all_docs_tokens:
                    if term in tokens:
                        docs_containing_term += 1

                # IDF
                if docs_containing_term > 0:
                    idf = math.log(
                        (total_docs - docs_containing_term + 0.5)
                        / (docs_containing_term + 0.5)
                    )
                else:
                    idf = 0.0

                # BM25 공식
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                score += idf * (numerator / denominator)

        # 0-1 범위로 정규화
        return min(score / len(query_tokens), 1.0)
