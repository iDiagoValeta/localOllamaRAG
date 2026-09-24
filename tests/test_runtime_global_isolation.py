"""Regression coverage for isolation of mutable rag.chat_pdfs globals."""

import rag.chat_pdfs as engine
from monkeygrab.config.app_config import AppConfig


_MUTATED_ROLES = {
    "rag": "mutated-rag",
    "chat": "mutated-chat",
    "contextual": "mutated-contextual",
    "recomp": "mutated-recomp",
}


def test_01_mutator_changes_runtime_configuration():
    engine.set_docs_folder_runtime("/tmp/mutated-docs")
    engine.set_model_roles_runtime(_MUTATED_ROLES)
    for name in engine.PIPELINE_RUNTIME_FLAGS:
        setattr(engine, name, not getattr(engine, name))
    for name in ("USAR_CONTEXTUAL_RETRIEVAL", "USAR_EMBEDDINGS_IMAGEN", "USAR_DESCRIPCION_IMAGEN"):
        setattr(engine, name, not getattr(engine, name))
    engine.LOGGING_METRICAS = not engine.LOGGING_METRICAS
    engine.GUARDAR_DEBUG_RAG = not engine.GUARDAR_DEBUG_RAG
    engine.CHUNK_SIZE = 12345
    engine.MODEL_ROLE_VARS["rag"] = "BROKEN_RAG_VARIABLE"


def test_02_observer_sees_the_previous_test_defaults():
    assert engine.CARPETA_DOCS == engine._DEFAULT_CARPETA_DOCS
    assert engine.PATH_DB == engine._DEFAULT_PATH_DB
    assert engine.COLLECTION_NAME == engine._DEFAULT_COLLECTION_NAME
    assert engine.get_model_roles() == engine._DEFAULT_MODEL_ROLES
    assert engine.MODELO_DESC == engine._inferir_descripcion_modelo(
        engine._DEFAULT_MODEL_ROLES["rag"]
    )
    assert engine.get_pipeline_flags() == {
        name: engine._DEFAULT_PIPELINE_FLAGS[name]
        for name in engine.PIPELINE_RUNTIME_FLAGS
    }
    assert engine.USAR_CONTEXTUAL_RETRIEVAL == engine._DEFAULT_PIPELINE_FLAGS[
        "USAR_CONTEXTUAL_RETRIEVAL"
    ]
    assert engine.USAR_EMBEDDINGS_IMAGEN == engine._DEFAULT_PIPELINE_FLAGS[
        "USAR_EMBEDDINGS_IMAGEN"
    ]
    assert engine.USAR_DESCRIPCION_IMAGEN == engine._DEFAULT_PIPELINE_FLAGS[
        "USAR_DESCRIPCION_IMAGEN"
    ]
    assert engine.LOGGING_METRICAS == AppConfig.from_env().flags.logging_metricas
    assert engine.GUARDAR_DEBUG_RAG == AppConfig.from_env().flags.guardar_debug_rag
    assert engine.CHUNK_SIZE == AppConfig.from_env().chunking.chunk_size
    assert engine.MODEL_ROLE_VARS == {
        "rag": "MODELO_RAG",
        "chat": "MODELO_CHAT",
        "contextual": "MODELO_CONTEXTUAL",
        "recomp": "MODELO_RECOMP",
    }
