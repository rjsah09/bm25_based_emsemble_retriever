import os
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain.schema import Document

"""
작성자: 양건모
최초 생성일: 2025.06.02
-----------------------
변경 내역
2025.06.02 주석 추가
"""


class SlideVectorDB:
    def __init__(
        self, persist_directory="./chroma_db/slides", collection_name="ppt_slides"
    ):
        """변수 및 관련 모듈 초기화"""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="intfloat/multilingual-e5-large"
            )
        )
        self.client = PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=self.embedding_function
        )

    def _extract_text_from_slide(self, slide_data):
        """pptx-parser 모듈 통해 슬라이드 단위로 추출된 데이터를 입력받아 연결된 텍스트로 변환"""

        def extract_text_recursive(items):
            texts = []
            for item in items:
                if item["content_type"] == "text":
                    texts.append(item["content"])
                elif item["content_type"] == "group":
                    texts.extend(extract_text_recursive(item["content"]))
            return texts

        return "\n".join(extract_text_recursive(slide_data["content_items"]))

    def create_documents(self, slides):
        """document 생성"""
        documents = []
        for slide in slides:
            text_content = self._extract_text_from_slide(slide)
            metadata = {
                "file_name": slide["file_name"],
                "slide_number": slide["slide_number"],
            }
            documents.append(Document(page_content=text_content, metadata=metadata))
        return documents

    def index_slides(self, slides):
        """슬라이드 인덱싱"""
        documents = self.create_documents(slides)
        ids = [
            f"{d.metadata['file_name']}_slide_{d.metadata['slide_number']}"
            for d in documents
        ]
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)
        print(
            f"'{self.collection_name}' 컬렉션에 {len(documents)}개 데이터 인덱싱 완료."
        )

    def query_slides(self, query, k=3):
        """Query"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        slides = []
        for i in range(len(results["documents"][0])):
            # 거리를 유사도로 변환 (거리가 작을수록 유사도가 높음)
            distance = results["distances"][0][i] if results["distances"] else 0.0
            similarity = max(0.0, 1.0 - distance)  # 거리를 유사도로 변환

            slides.append(
                {
                    "slide_number": results["metadatas"][0][i]["slide_number"],
                    "file_name": results["metadatas"][0][i]["file_name"],
                    "text_content": results["documents"][0][i],
                    "similarity": similarity,
                }
            )
        return slides

    def delete_slides_by_file(self, file_name: str):
        """파일명으로 모든 슬라이드 데이터 삭제"""
        results = self.collection.get(include=["metadatas", "ids"])
        delete_ids = [
            doc_id
            for doc_id, meta in zip(results["ids"], results["metadatas"])
            if meta.get("file_name") == file_name
        ]
        if delete_ids:
            self.collection.delete(ids=delete_ids)
            print(f"Deleted {len(delete_ids)} slides for file '{file_name}'")
        else:
            print(f"No slides found for deletion in file '{file_name}'")

    def update_slides(self, slides):
        """슬라이드 업데이트"""
        file_name = slides[0]["file_name"]
        self.delete_slides_by_file(file_name)
        self.index_slides(slides)
