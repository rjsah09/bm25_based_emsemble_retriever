class EnsembleRetriever:
    def __init__(self, retrievers: list[dict]):
        self.retrievers = retrievers

    def retrieve(self, query: str):
        # 각 retriever에서 검색 결과 수집
        all_results = []

        for retriever_config in self.retrievers:
            retriever = retriever_config["retriever"]
            weight = retriever_config["weight"]

            # 각 retriever에서 검색 수행
            results = retriever.retrieve(query)
            print(results)

            # 가중치를 각 결과에 적용
            for result in results:
                result_copy = result.copy()
                result_copy["weight"] = weight
                all_results.append(result_copy)

        unique_results = {}
        for result in all_results:
            file_name = result.get("file_name", "")
            slide_index = result.get("slide_index", -1)
            key = (file_name, slide_index)

            if key not in unique_results:
                unique_results[key] = result
            else:
                # 기존 결과와 가중치 및 유사도 합산
                existing_weight = unique_results[key].get("weight", 0.0)
                new_weight = result.get("weight", 0.0)
                unique_results[key]["weight"] = existing_weight + new_weight

                # 가중 평균으로 유사도 계산
                existing_similarity = unique_results[key].get("similarity", 0.0)
                new_similarity = result.get("similarity", 0.0)
                total_weight = existing_weight + new_weight

                if total_weight > 0:
                    weighted_similarity = (
                        existing_similarity * existing_weight
                        + new_similarity * new_weight
                    ) / total_weight
                    unique_results[key]["similarity"] = weighted_similarity

        # 가중치 기준으로 정렬 (높은 가중치 순)
        sorted_results = sorted(
            unique_results.values(), key=lambda x: x.get("weight", 0.0), reverse=True
        )

        # 상위 k개 결과 반환
        return sorted_results[:k]

    def get(self, query: str, k: int = 50):
        pass  # TODO: 추후 구현 예정

    def save(self, path: str):
        for retriever in self.retrievers:
            retriever.save(path)

    def save_from_path(self, path: str):
        for retriever in self.retrievers:
            retriever.save_from_path(path)

    def save_from_external_path(self, path: str):
        for retriever in self.retrievers:
            retriever.save_from_external_path(path)

    def delete_all(self):
        for retriever in self.retrievers:
            retriever.delete_all()

    def delete_by_file_name(self, file_name: str):
        for retriever in self.retrievers:
            retriever.delete_by_file_name(file_name)
