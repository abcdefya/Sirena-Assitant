def test_pipeline_compiles_without_error():
    from unittest.mock import patch, MagicMock
    with patch("src.nodes.decide_node.get_llm", return_value=MagicMock()), \
         patch("src.nodes.retrieve_node.get_retriever", return_value=MagicMock()), \
         patch("src.nodes.grade_node.get_llm", return_value=MagicMock()):
        from src.agents import pipeline as pipeline_module
        pipeline_module._app = None
        app = pipeline_module.get_pipeline()
        assert app is not None


def test_pipeline_is_singleton():
    from src.agents import pipeline as pipeline_module
    pipeline_module._app = None
    app1 = pipeline_module.get_pipeline()
    app2 = pipeline_module.get_pipeline()
    assert app1 is app2
