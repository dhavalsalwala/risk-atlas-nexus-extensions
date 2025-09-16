import os
from uuid import uuid4

from linkml_runtime.dumpers import YAMLDumper

from ran_ares_integration.assets import DATA_DIR
from ran_ares_integration.datamodel.risk_to_ares_ontology import (
    ARESConfig,
    AresEvaluator,
    ARESGoal,
    AresIntent,
    ARESStrategy,
    ChatTemplate,
    Connector,
    GenerateKwargs,
    GenerateParams,
    ModelConfig,
    RedTeaming,
    RiskToARES,
    RiskToARESMapping,
    TokenizerConfig,
)


direct_instructions_attack = RiskToARESMapping(
    **{
        "id": str(uuid4()),
        "name": "direct_instructions_attack",
        "description": None,
        "risk_id": "direct-instructions-attack",
        "ares_config": ARESConfig(
            **{
                "id": str(uuid4()),
                "name": "direct_instructions_attack_config",
                "description": None,
                "target": {
                    "huggingface": Connector(
                        id=str(uuid4()),
                        type="ares.connectors.huggingface.HuggingFaceConnector",
                        model_configs=ModelConfig(
                            pretrained_model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
                            torch_dtype="bfloat16",
                        ),
                        tokenizer_config=TokenizerConfig(
                            pretrained_model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
                            padding_side="left",
                        ),
                        generate_kwargs=GenerateKwargs(
                            chat_template=ChatTemplate(
                                return_tensors="pt",
                                thinking=True,
                                return_dict=True,
                                add_generation_prompt=True,
                            ),
                            generate_params=GenerateParams(max_new_tokens=50),
                        ),
                        seed=42,
                        device="auto",
                    )
                },
                "red_teaming": RedTeaming(
                    **{
                        "id": str(uuid4()),
                        "description": None,
                        "intent": "owasp-llm-02",
                        "intent_config": {
                            "owasp-llm-02": AresIntent(
                                id=str(uuid4()),
                                description=None,
                                goal=ARESGoal(
                                    **{
                                        "id": str(uuid4()),
                                        "name": "Generic Attack Goal",
                                        "type": "ares.goals.generic_attack_goal.GenericAttackGoal",
                                        "output_path": "results/assets/attack_goals_output.json",
                                        "goal": "Behavior",
                                    }
                                ),
                                strategy={
                                    "direct_requests": ARESStrategy(
                                        **{
                                            "id": str(uuid4()),
                                            "name": "Direct Requests",
                                            "type": "ares.strategies.direct_requests.DirectRequests",
                                            "input_path": "results/assets/attack_goals_output.json",
                                            "output_path": "results/assets/direct_requests_output.json",
                                        }
                                    )
                                },
                                evaluation=AresEvaluator(
                                    **{
                                        "id": str(uuid4()),
                                        "name": "keyword",
                                        "type": "ares.evals.keyword_eval.KeywordEval",
                                        "keyword_list_or_path": "ares_assets/advbench_refusal_keywords.json",
                                    }
                                ),
                            )
                        },
                        "prompts": "seeds.csv",
                    }
                ),
            }
        ),
    }
)

encoded_interactions_attack = RiskToARESMapping(
    **{
        "id": str(uuid4()),
        "name": "encoded_interactions_attack",
        "description": None,
        "risk_id": "encoded-interactions-attack",
        "ares_config": ARESConfig(
            **{
                "id": str(uuid4()),
                "name": "encoded_interactions_attack_config",
                "description": None,
                "target": {
                    "huggingface": Connector(
                        id=str(uuid4()),
                        type="ares.connectors.huggingface.HuggingFaceConnector",
                        model_configs=ModelConfig(
                            pretrained_model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
                            torch_dtype="bfloat16",
                        ),
                        tokenizer_config=TokenizerConfig(
                            pretrained_model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
                            padding_side="left",
                        ),
                        generate_kwargs=GenerateKwargs(
                            chat_template=ChatTemplate(
                                return_tensors="pt",
                                thinking=True,
                                return_dict=True,
                                add_generation_prompt=True,
                            ),
                            generate_params=GenerateParams(max_new_tokens=50),
                        ),
                        seed=42,
                        device="auto",
                    )
                },
                "red_teaming": RedTeaming(
                    **{
                        "id": str(uuid4()),
                        "description": None,
                        "intent": "owasp-llm-02",
                        "intent_config": {
                            "owasp-llm-02": AresIntent(
                                id=str(uuid4()),
                                description=None,
                                goal=ARESGoal(
                                    **{
                                        "id": str(uuid4()),
                                        "name": "Generic Attack Goal",
                                        "type": "ares.goals.generic_attack_goal.GenericAttackGoal",
                                        "output_path": "results/assets/attack_goals_output.json",
                                        "goal": "Behavior",
                                    }
                                ),
                                strategy={
                                    "probes.encoding.InjectROT13": ARESStrategy(
                                        **{
                                            "id": str(uuid4()),
                                            "name": "Inject ASCII",
                                            "type": "ares_garak.strategies.encoding.EncodingStrategy",
                                            "probe": "probes.encoding.InjectROT13",
                                            "input_path": "results/assets/attack_goals_output.json",
                                            "output_path": "results/assets/garak_InjectROT13_output.json",
                                        }
                                    ),
                                },
                                evaluation=AresEvaluator(
                                    **{
                                        "id": str(uuid4()),
                                        "name": "keyword",
                                        "type": "ares.evals.keyword_eval.KeywordEval",
                                        "keyword_list_or_path": "ares_assets/advbench_refusal_keywords.json",
                                    }
                                ),
                            )
                        },
                        "prompts": "seeds.csv",
                    }
                ),
            }
        ),
    }
)

social_hacking_attack = RiskToARESMapping(
    **{
        "id": str(uuid4()),
        "name": "social_hacking_attack",
        "description": None,
        "risk_id": "social-hacking-attack",
        "ares_config": ARESConfig(
            **{
                "id": str(uuid4()),
                "name": "social_hacking_attack_config",
                "description": None,
                "target": {
                    "huggingface": Connector(
                        id=str(uuid4()),
                        type="ares.connectors.huggingface.HuggingFaceConnector",
                        model_configs=ModelConfig(
                            pretrained_model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
                            torch_dtype="bfloat16",
                        ),
                        tokenizer_config=TokenizerConfig(
                            pretrained_model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
                            padding_side="left",
                        ),
                        generate_kwargs=GenerateKwargs(
                            chat_template=ChatTemplate(
                                return_tensors="pt",
                                thinking=True,
                                return_dict=True,
                                add_generation_prompt=True,
                            ),
                            generate_params=GenerateParams(max_new_tokens=50),
                        ),
                        seed=42,
                        device="auto",
                    )
                },
                "red_teaming": RedTeaming(
                    **{
                        "id": str(uuid4()),
                        "description": None,
                        "intent": "owasp-llm-02",
                        "intent_config": {
                            "owasp-llm-02": AresIntent(
                                id=str(uuid4()),
                                description=None,
                                goal=ARESGoal(
                                    **{
                                        "id": str(uuid4()),
                                        "name": "Generic Attack Goal",
                                        "type": "ares.goals.generic_attack_goal.GenericAttackGoal",
                                        "output_path": "results/assets/attack_goals_output.json",
                                        "goal": "Behavior",
                                    }
                                ),
                                strategy={
                                    "human_jailbreak": ARESStrategy(
                                        **{
                                            "id": str(uuid4()),
                                            "name": "Human Jailbreak",
                                            "type": "ares_human_jailbreak.strategies.human_jailbreak.HumanJailbreak",
                                            "jailbreaks_path": "assets/human_jailbreaks.json",
                                            "input_path": "results/assets/attack_goals_output.json",
                                            "output_path": "results/assets/human_jailbreak_output.json",
                                        }
                                    ),
                                },
                                evaluation=AresEvaluator(
                                    **{
                                        "id": str(uuid4()),
                                        "name": "keyword",
                                        "type": "ares.evals.keyword_eval.KeywordEval",
                                        "keyword_list_or_path": "ares_assets/advbench_refusal_keywords.json",
                                    }
                                ),
                            )
                        },
                        "prompts": "seeds.csv",
                    }
                ),
            }
        ),
    }
)

specialized_tokens_attack = RiskToARESMapping(
    **{
        "id": str(uuid4()),
        "name": "specialized_tokens_attack",
        "description": None,
        "risk_id": "specialized-tokens-attack",
        "ares_config": ARESConfig(
            **{
                "id": str(uuid4()),
                "name": "specialized_tokens_attack_config",
                "description": None,
                "target": {
                    "huggingface": Connector(
                        id=str(uuid4()),
                        type="ares.connectors.huggingface.HuggingFaceConnector",
                        model_configs=ModelConfig(
                            pretrained_model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
                            torch_dtype="bfloat16",
                        ),
                        tokenizer_config=TokenizerConfig(
                            pretrained_model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
                            padding_side="left",
                        ),
                        generate_kwargs=GenerateKwargs(
                            chat_template=ChatTemplate(
                                return_tensors="pt",
                                thinking=True,
                                return_dict=True,
                                add_generation_prompt=True,
                            ),
                            generate_params=GenerateParams(max_new_tokens=50),
                        ),
                        seed=42,
                        device="auto",
                    )
                },
                "red_teaming": RedTeaming(
                    **{
                        "id": str(uuid4()),
                        "description": None,
                        "intent": "owasp-llm-02",
                        "intent_config": {
                            "owasp-llm-02": AresIntent(
                                id=str(uuid4()),
                                description=None,
                                goal=ARESGoal(
                                    **{
                                        "id": str(uuid4()),
                                        "name": "Generic Attack Goal",
                                        "type": "ares.goals.generic_attack_goal.GenericAttackGoal",
                                        "output_path": "results/assets/attack_goals_output.json",
                                        "goal": "Behavior",
                                    }
                                ),
                                strategy={
                                    "direct_requests": ARESStrategy(
                                        **{
                                            "id": str(uuid4()),
                                            "name": "Direct Requests",
                                            "type": "ares.strategies.direct_requests.DirectRequests",
                                            "input_path": "results/assets/attack_goals_output.json",
                                            "output_path": "results/assets/direct_requests_output.json",
                                        }
                                    ),
                                    "human_jailbreak": ARESStrategy(
                                        **{
                                            "id": str(uuid4()),
                                            "name": "Human Jailbreak",
                                            "type": "ares_human_jailbreak.strategies.human_jailbreak.HumanJailbreak",
                                            "jailbreaks_path": "assets/human_jailbreaks.json",
                                            "input_path": "results/assets/attack_goals_output.json",
                                            "output_path": "results/assets/human_jailbreak_output.json",
                                        }
                                    ),
                                    "probes.encoding.InjectROT13": ARESStrategy(
                                        **{
                                            "id": str(uuid4()),
                                            "name": "Inject ASCII",
                                            "type": "ares_garak.strategies.encoding.EncodingStrategy",
                                            "probe": "probes.encoding.InjectROT13",
                                            "input_path": "results/assets/attack_goals_output.json",
                                            "output_path": "results/assets/garak_InjectROT13_output.json",
                                        }
                                    ),
                                },
                                evaluation=AresEvaluator(
                                    **{
                                        "id": str(uuid4()),
                                        "name": "keyword",
                                        "type": "ares.evals.keyword_eval.KeywordEval",
                                        "keyword_list_or_path": "ares_assets/advbench_refusal_keywords.json",
                                    }
                                ),
                            )
                        },
                        "prompts": "assets/pii-seeds.csv",
                    }
                ),
            }
        ),
    }
)

d = RiskToARES(
    mappings=[
        direct_instructions_attack,
        encoded_interactions_attack,
        social_hacking_attack,
        specialized_tokens_attack,
    ]
).model_json_schema(mode="serialization")
d


with open(
    os.path.join(DATA_DIR, "knowledge_graph", "risk_to_ares_mappings.yaml"),
    "+tw",
    encoding="utf-8",
) as output_file:
    print(
        YAMLDumper().dumps(
            RiskToARES(
                mappings=[
                    direct_instructions_attack,
                    encoded_interactions_attack,
                    social_hacking_attack,
                    specialized_tokens_attack,
                ]
            )
        ),
        file=output_file,
    )
