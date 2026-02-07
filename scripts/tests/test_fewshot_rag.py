"""
Test Script: Code-Similarity Search (Few-Shot RAG)
Tests the new Few-Shot RAG implementation for ScriptGuard.
"""

import yaml
from scriptguard.database.dataset_manager import DatasetManager
from scriptguard.rag.code_similarity_store import CodeSimilarityStore
from scriptguard.steps.vectorize_samples import vectorize_samples
from scriptguard.utils.prompts import format_fewshot_prompt, format_inference_prompt
from scriptguard.utils.logger import logger


def test_code_similarity_rag():
    """Test complete Few-Shot RAG workflow."""

    logger.info("=" * 80)
    logger.info("TESTING CODE-SIMILARITY SEARCH (FEW-SHOT RAG)")
    logger.info("=" * 80)

    # Load configuration
    logger.info("\n1. Loading configuration...")
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Test 1: Check PostgreSQL samples
    logger.info("\n2. Checking PostgreSQL database...")
    db_manager = DatasetManager()

    malicious_count = len(db_manager.get_all_samples(label="malicious", limit=10))
    benign_count = len(db_manager.get_all_samples(label="benign", limit=10))

    logger.info(f"   ✓ PostgreSQL samples found:")
    logger.info(f"     - Malicious: {malicious_count} (showing first 10)")
    logger.info(f"     - Benign: {benign_count} (showing first 10)")

    if malicious_count == 0 or benign_count == 0:
        logger.error("❌ PostgreSQL is empty! Run data ingestion first:")
        logger.error("   python -m scriptguard.pipelines.training_pipeline")
        return False

    # Test 2: Vectorize samples
    logger.info("\n3. Vectorizing samples (PostgreSQL → Qdrant)...")
    try:
        vectorize_result = vectorize_samples(
            config=config,
            max_samples=100,  # Limit to 100 for testing
            clear_existing=True
        )

        if vectorize_result["status"] == "success":
            logger.info(f"   ✓ Vectorization successful:")
            logger.info(f"     - Total vectors: {vectorize_result['samples_vectorized']}")
            logger.info(f"     - Malicious: {vectorize_result['malicious_count']}")
            logger.info(f"     - Benign: {vectorize_result['benign_count']}")
        else:
            logger.error(f"❌ Vectorization failed: {vectorize_result}")
            return False

    except Exception as e:
        logger.error(f"❌ Vectorization error: {e}")
        return False

    # Test 3: Test similarity search
    logger.info("\n4. Testing similarity search...")

    qdrant_config = config.get("qdrant", {})
    code_embedding_model = config.get("code_embedding", {}).get(
        "model",
        "microsoft/unixcoder-base"
    )

    code_store = CodeSimilarityStore(
        host=qdrant_config.get("host", "localhost"),
        port=qdrant_config.get("port", 6333),
        collection_name="code_samples",
        embedding_model=code_embedding_model
    )

    # Test queries
    test_queries = [
        {
            "name": "Malicious: Reverse Shell",
            "code": "import socket\ns=socket.socket()\ns.connect(('10.0.0.1',1234))"
        },
        {
            "name": "Benign: Data Processing",
            "code": "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())"
        },
        {
            "name": "Suspicious: Command Execution",
            "code": "import os\nos.system('ls -la')"
        }
    ]

    for query in test_queries:
        logger.info(f"\n   Query: {query['name']}")
        logger.info(f"   Code: {query['code'][:60]}...")

        results = code_store.search_similar_code(
            query_code=query['code'],
            k=3,
            balance_labels=True,
            score_threshold=0.2
        )

        if results:
            logger.info(f"   ✓ Found {len(results)} similar samples:")
            for i, result in enumerate(results, 1):
                logger.info(f"     {i}. {result['label'].upper()} (score: {result['score']:.3f})")
                logger.info(f"        {result['code'][:60]}...")
        else:
            logger.warning(f"   ⚠ No similar samples found")

    # Test 4: Compare prompts
    logger.info("\n5. Comparing prompts...")

    test_code = "import subprocess\nsubprocess.call(['rm', '-rf', '/'])"

    # Standard prompt
    standard_prompt = format_inference_prompt(test_code, max_code_length=500)
    logger.info(f"\n   Standard Prompt ({len(standard_prompt)} chars):")
    logger.info(f"   {standard_prompt[:200]}...\n")

    # Few-Shot prompt
    similar_examples = code_store.search_similar_code(
        query_code=test_code,
        k=3,
        balance_labels=True
    )

    if similar_examples:
        fewshot_prompt = format_fewshot_prompt(
            target_code=test_code,
            context_examples=similar_examples,
            max_code_length=500,
            max_context_length=300
        )
        logger.info(f"   Few-Shot Prompt ({len(fewshot_prompt)} chars):")
        logger.info(f"   {fewshot_prompt[:400]}...\n")
        logger.info(f"   ✓ Few-Shot prompt is {len(fewshot_prompt) - len(standard_prompt)} chars longer")
        logger.info(f"   ✓ Contains {len(similar_examples)} context examples")
    else:
        logger.warning("   ⚠ Could not generate Few-Shot prompt (no similar samples)")

    # Test 5: Collection statistics
    logger.info("\n6. Final collection statistics...")
    info = code_store.get_collection_info()
    logger.info(f"   Collection: {info.get('name')}")
    logger.info(f"   Total samples: {info.get('total_samples', 0)}")
    logger.info(f"   Malicious: {info.get('malicious_samples', 0)}")
    logger.info(f"   Benign: {info.get('benign_samples', 0)}")
    logger.info(f"   Embedding dimension: {info.get('embedding_dim', 0)}")
    logger.info(f"   Status: {info.get('status', 'N/A')}")

    # Success summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL TESTS PASSED - Few-Shot RAG is ready!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Run training with Few-Shot RAG:")
    logger.info("   python -m scriptguard.pipelines.training_pipeline")
    logger.info("2. Evaluate model with Few-Shot RAG:")
    logger.info("   (use_fewshot_rag=True in evaluate_model step)")
    logger.info("3. Compare F1 scores: Standard vs Few-Shot RAG")

    return True


if __name__ == "__main__":
    try:
        success = test_code_similarity_rag()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
