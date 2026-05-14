"""AWS Bedrock judge configurator for RAGAS.

Required:
    pip install langchain-aws boto3

AWS credentials via any of:
    - AWS_BEARER_TOKEN_BEDROCK  (recommended Bedrock long-term API key)
    - IAM credentials (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_DEFAULT_REGION)
    - ~/.aws/credentials (aws configure)
    - profile name passed to ``build_aws_configurator``.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Callable

DEFAULT_CHAT_MODEL = "eu.anthropic.claude-sonnet-4-20250514-v1:0"
DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_REGION = "eu-north-1"
DEFAULT_MAX_TOKENS = 4096


def build_aws_configurator(args: argparse.Namespace) -> Callable:
    """Return a ``(google_timeout, google_retries) -> (llm, embeddings)`` callable.

    The returned configurator is passed to ``evaluar_respuestas_con_ragas`` via
    its ``llm_configurator`` parameter. The Google-style signature is kept for
    backward compatibility with the RAGAS runner; the timeout/retries args are
    ignored by Bedrock providers.
    """
    def configurar_llm_aws(google_timeout=None, google_retries=None):
        try:
            from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
            from ragas.llms.base import LangchainLLMWrapper
        except ImportError as err:
            print(f"Error: {err}")
            print("Install with: pip install langchain-aws boto3")
            raise SystemExit(1) from err

        if args.aws_profile:
            import boto3
            boto3.setup_default_session(
                region_name=args.aws_region,
                profile_name=args.aws_profile,
            )

        raw_eval_llm = ChatBedrockConverse(
            model=args.aws_model,
            region_name=args.aws_region,
            temperature=args.aws_temperature,
            max_tokens=args.aws_max_tokens,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="LangchainLLMWrapper is deprecated.*",
                category=DeprecationWarning,
            )
            eval_llm = LangchainLLMWrapper(raw_eval_llm, bypass_n=True)

        eval_embeddings = None
        if args.aws_embedding_model.lower() != "none":
            eval_embeddings = BedrockEmbeddings(
                model_id=args.aws_embedding_model,
                region_name=args.aws_region,
            )

        print(f"Evaluation LLM:        AWS Bedrock {args.aws_model}")
        print(
            "Evaluation embeddings: "
            + ("disabled" if eval_embeddings is None else f"AWS Bedrock {args.aws_embedding_model}")
        )
        print(f"Region: {args.aws_region}" + (f"  profile: {args.aws_profile}" if args.aws_profile else ""))
        print(
            "RAGAS throughput config: "
            f"workers={args.ragas_max_workers}, "
            f"batch_size={args.ragas_batch_size or 'auto'}"
        )
        return eval_llm, eval_embeddings

    return configurar_llm_aws


def add_aws_args(parser: argparse.ArgumentParser) -> None:
    """Attach AWS-specific CLI flags to an argparse parser."""
    parser.add_argument("--aws-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--aws-embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--aws-region", default=DEFAULT_REGION)
    parser.add_argument("--aws-profile", default=None, help="AWS credentials profile name.")
    parser.add_argument("--aws-temperature", type=float, default=0.0)
    parser.add_argument("--aws-max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
