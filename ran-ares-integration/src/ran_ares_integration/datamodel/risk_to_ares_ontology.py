from __future__ import annotations

import re
import sys
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator


metamodel_version = "None"
version = "0.0.1"


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        validate_default=True,
        extra="allow",
        arbitrary_types_allowed=True,
        use_enum_values=True,
        strict=False,
    )
    pass


class LinkMLMeta(RootModel):
    root: dict[str, Any] = {}
    model_config = ConfigDict(frozen=True)

    def __getattr__(self, key: str):
        return getattr(self.root, key)

    def __getitem__(self, key: str):
        return self.root[key]

    def __setitem__(self, key: str, value):
        self.root[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.root


linkml_meta = LinkMLMeta(
    {
        "default_curi_maps": ["semweb_context"],
        "default_prefix": "https://ibm.github.io/ran-ares-integration/ontology/risk_to_ares/",
        "default_range": "string",
        "description": "Vocabulary to integrate Ares workflow",
        "id": "https://ibm.github.io/ran-ares-integration/ontology/risk_to_ares",
        "imports": ["linkml:types", "common", "ares_config"],
        "name": "ares",
        "prefixes": {
            "linkml": {
                "prefix_prefix": "linkml",
                "prefix_reference": "https://w3id.org/linkml/",
            }
        },
        "source_file": "risk_to_ares.yaml",
    }
)


class Entity(ConfiguredBaseModel):
    """
    A generic grouping for any identifiable entity.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "abstract": True,
            "class_uri": "schema:Thing",
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/common",
        }
    )

    id: str = Field(
        default=...,
        description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "id",
                "domain_of": ["Entity"],
                "slot_uri": "schema:identifier",
            }
        },
    )
    name: Optional[str] = Field(
        default=None,
        description="""A text name of this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "name",
                "domain_of": ["Entity"],
                "slot_uri": "schema:name",
            }
        },
    )
    description: Optional[str] = Field(
        default=None,
        description="""The description of an entity""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "description",
                "domain_of": ["Entity"],
                "slot_uri": "schema:description",
            }
        },
    )
    url: Optional[str] = Field(
        default=None,
        description="""An optional URL associated with this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "url",
                "domain_of": ["Entity"],
                "slot_uri": "schema:url",
            }
        },
    )
    dateCreated: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was created.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateCreated",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateCreated",
            }
        },
    )
    dateModified: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was most recently modified.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateModified",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateModified",
            }
        },
    )


class ARESGoal(Entity):
    """
    Base ARES Goal
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/ares_config"
        }
    )

    type: str = Field(
        default=...,
        description="""String describing the python type""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "type",
                "domain_of": ["ARESGoal", "ARESStrategy", "AresEvaluator", "Connector"],
            }
        },
    )
    origin: str = Field(
        default="local",
        description="""local or remote""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "origin",
                "domain_of": ["ARESGoal"],
                "ifabsent": "local",
            }
        },
    )
    base_path: Optional[str] = Field(
        default=None,
        description="""path to input file""",
        json_schema_extra={
            "linkml_meta": {"alias": "base_path", "domain_of": ["ARESGoal"]}
        },
    )
    output_path: str = Field(
        default=...,
        description="""filename to output of the processed goals""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "output_path",
                "domain_of": ["ARESGoal", "ARESStrategy", "AresEvaluator"],
            }
        },
    )
    goal: Optional[str] = Field(
        default=None,
        description="""column name of the field in the input file to be used as source of goals""",
        json_schema_extra={
            "linkml_meta": {"alias": "goal", "domain_of": ["ARESGoal", "AresIntent"]}
        },
    )
    builder_kwargs: Optional[str] = Field(
        default=None,
        description="""column name of the field in the input file to be used as source of labels.""",
        json_schema_extra={
            "linkml_meta": {"alias": "builder_kwargs", "domain_of": ["ARESGoal"]}
        },
    )
    task_kwargs: Optional[str] = Field(
        default=None,
        description="""column name of the field in the input file to be used as source of labels.""",
        json_schema_extra={
            "linkml_meta": {"alias": "task_kwargs", "domain_of": ["ARESGoal"]}
        },
    )
    base_kwargs: Optional[str] = Field(
        default=None,
        description="""column name of the field in the input file to be used as source of labels.""",
        json_schema_extra={
            "linkml_meta": {"alias": "base_kwargs", "domain_of": ["ARESGoal"]}
        },
    )
    id: str = Field(
        default=...,
        description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "id",
                "domain_of": ["Entity"],
                "slot_uri": "schema:identifier",
            }
        },
    )
    name: Optional[str] = Field(
        default=None,
        description="""A text name of this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "name",
                "domain_of": ["Entity"],
                "slot_uri": "schema:name",
            }
        },
    )
    description: Optional[str] = Field(
        default=None,
        description="""The description of an entity""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "description",
                "domain_of": ["Entity"],
                "slot_uri": "schema:description",
            }
        },
    )
    url: Optional[str] = Field(
        default=None,
        description="""An optional URL associated with this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "url",
                "domain_of": ["Entity"],
                "slot_uri": "schema:url",
            }
        },
    )
    dateCreated: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was created.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateCreated",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateCreated",
            }
        },
    )
    dateModified: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was most recently modified.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateModified",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateModified",
            }
        },
    )


class ARESStrategy(Entity):
    """
    The type of attacks or evaluation techniques.  The strategy used for red-teaming the language model and, in particular, for transforming the goal prompts saved in the previous step to adversarial attack prompts.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/ares_config"
        }
    )

    type: str = Field(
        default=...,
        description="""String describing the python type""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "type",
                "domain_of": ["ARESGoal", "ARESStrategy", "AresEvaluator", "Connector"],
            }
        },
    )
    input_path: str = Field(
        default=...,
        description="""The input path""",
        json_schema_extra={
            "linkml_meta": {"alias": "input_path", "domain_of": ["ARESStrategy"]}
        },
    )
    output_path: str = Field(
        default=...,
        description="""The output path""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "output_path",
                "domain_of": ["ARESGoal", "ARESStrategy", "AresEvaluator"],
            }
        },
    )
    jailbreaks_path: Optional[str] = Field(
        default=None,
        description="""String describing the python type""",
        json_schema_extra={
            "linkml_meta": {"alias": "jailbreaks_path", "domain_of": ["ARESStrategy"]}
        },
    )
    probe: Optional[str] = Field(
        default=None,
        description="""String describing the probe type""",
        json_schema_extra={
            "linkml_meta": {"alias": "probe", "domain_of": ["ARESStrategy"]}
        },
    )
    templates: Optional[str] = Field(
        default=None,
        description="""String describing the python type""",
        json_schema_extra={
            "linkml_meta": {"alias": "templates", "domain_of": ["ARESStrategy"]}
        },
    )
    id: str = Field(
        default=...,
        description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "id",
                "domain_of": ["Entity"],
                "slot_uri": "schema:identifier",
            }
        },
    )
    name: Optional[str] = Field(
        default=None,
        description="""A text name of this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "name",
                "domain_of": ["Entity"],
                "slot_uri": "schema:name",
            }
        },
    )
    description: Optional[str] = Field(
        default=None,
        description="""The description of an entity""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "description",
                "domain_of": ["Entity"],
                "slot_uri": "schema:description",
            }
        },
    )
    url: Optional[str] = Field(
        default=None,
        description="""An optional URL associated with this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "url",
                "domain_of": ["Entity"],
                "slot_uri": "schema:url",
            }
        },
    )
    dateCreated: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was created.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateCreated",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateCreated",
            }
        },
    )
    dateModified: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was most recently modified.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateModified",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateModified",
            }
        },
    )


class AresEvaluator(Entity):
    """
    How to measure and report results. The evaluation node describes the evaluators assessing the performance of the target LM under attack.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/ares_config"
        }
    )

    type: str = Field(
        default=...,
        description="""String describing the python type""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "type",
                "domain_of": ["ARESGoal", "ARESStrategy", "AresEvaluator", "Connector"],
            }
        },
    )
    output_path: str = Field(
        default="results/evaluation.json",
        description="""The output path for the evaluation results""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "output_path",
                "domain_of": ["ARESGoal", "ARESStrategy", "AresEvaluator"],
                "ifabsent": "results/evaluation.json",
            }
        },
    )
    sensitive_type: Optional[str] = Field(
        default=None,
        description="""String describing the ARES evaluation type""",
        json_schema_extra={
            "linkml_meta": {"alias": "sensitive_type", "domain_of": ["AresEvaluator"]}
        },
    )
    exclude_prompt: Optional[bool] = Field(
        default=None,
        description="""The input path the path to dataset of attacks generated by strategy""",
        json_schema_extra={
            "linkml_meta": {"alias": "exclude_prompt", "domain_of": ["AresEvaluator"]}
        },
    )
    debug_mode: Optional[bool] = Field(
        default=None,
        description="""The output path for the evaluation results""",
        json_schema_extra={
            "linkml_meta": {"alias": "debug_mode", "domain_of": ["AresEvaluator"]}
        },
    )
    keyword_list_or_path: Optional[str] = Field(
        default=None,
        description="""String describing the ARES evaluation type""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "keyword_list_or_path",
                "domain_of": ["AresEvaluator"],
            }
        },
    )
    id: str = Field(
        default=...,
        description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "id",
                "domain_of": ["Entity"],
                "slot_uri": "schema:identifier",
            }
        },
    )
    name: Optional[str] = Field(
        default=None,
        description="""A text name of this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "name",
                "domain_of": ["Entity"],
                "slot_uri": "schema:name",
            }
        },
    )
    description: Optional[str] = Field(
        default=None,
        description="""The description of an entity""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "description",
                "domain_of": ["Entity"],
                "slot_uri": "schema:description",
            }
        },
    )
    url: Optional[str] = Field(
        default=None,
        description="""An optional URL associated with this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "url",
                "domain_of": ["Entity"],
                "slot_uri": "schema:url",
            }
        },
    )
    dateCreated: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was created.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateCreated",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateCreated",
            }
        },
    )
    dateModified: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was most recently modified.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateModified",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateModified",
            }
        },
    )


class AresIntent(Entity):
    """
    An ARES intent
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "class_uri": "schema:AresIntent",
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/ares_config",
        }
    )

    goal: ARESGoal = Field(
        default=...,
        json_schema_extra={
            "linkml_meta": {"alias": "goal", "domain_of": ["ARESGoal", "AresIntent"]}
        },
    )
    strategy: dict[str, ARESStrategy] = Field(
        default=...,
        description="""The path to the prompts file""",
        json_schema_extra={
            "linkml_meta": {"alias": "strategy", "domain_of": ["AresIntent"]}
        },
    )
    evaluation: AresEvaluator = Field(
        default=...,
        description="""The path to the prompts file""",
        json_schema_extra={
            "linkml_meta": {"alias": "evaluation", "domain_of": ["AresIntent"]}
        },
    )
    id: str = Field(
        default=...,
        description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "id",
                "domain_of": ["Entity"],
                "slot_uri": "schema:identifier",
            }
        },
    )
    name: Optional[str] = Field(
        default=None,
        description="""A text name of this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "name",
                "domain_of": ["Entity"],
                "slot_uri": "schema:name",
            }
        },
    )
    description: Optional[str] = Field(
        default=None,
        description="""The description of an entity""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "description",
                "domain_of": ["Entity"],
                "slot_uri": "schema:description",
            }
        },
    )
    url: Optional[str] = Field(
        default=None,
        description="""An optional URL associated with this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "url",
                "domain_of": ["Entity"],
                "slot_uri": "schema:url",
            }
        },
    )
    dateCreated: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was created.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateCreated",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateCreated",
            }
        },
    )
    dateModified: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was most recently modified.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateModified",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateModified",
            }
        },
    )


class RedTeaming(Entity):
    """
    ARES uses intents to configure red-teaming.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/ares_config"
        }
    )

    intent: str = Field(
        default=...,
        description="""intent name""",
        json_schema_extra={
            "linkml_meta": {"alias": "intent", "domain_of": ["RedTeaming"]}
        },
    )
    intent_config: dict[str, AresIntent] = Field(
        default=...,
        description="""intent configuration""",
        json_schema_extra={
            "linkml_meta": {"alias": "intent_config", "domain_of": ["RedTeaming"]}
        },
    )
    prompts: str = Field(
        default=...,
        description="""The path to the prompts file""",
        json_schema_extra={
            "linkml_meta": {"alias": "prompts", "domain_of": ["RedTeaming"]}
        },
    )
    id: str = Field(
        default=...,
        description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "id",
                "domain_of": ["Entity"],
                "slot_uri": "schema:identifier",
            }
        },
    )
    name: Optional[str] = Field(
        default=None,
        description="""A text name of this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "name",
                "domain_of": ["Entity"],
                "slot_uri": "schema:name",
            }
        },
    )
    description: Optional[str] = Field(
        default=None,
        description="""The description of an entity""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "description",
                "domain_of": ["Entity"],
                "slot_uri": "schema:description",
            }
        },
    )
    url: Optional[str] = Field(
        default=None,
        description="""An optional URL associated with this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "url",
                "domain_of": ["Entity"],
                "slot_uri": "schema:url",
            }
        },
    )
    dateCreated: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was created.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateCreated",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateCreated",
            }
        },
    )
    dateModified: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was most recently modified.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateModified",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateModified",
            }
        },
    )


class Connector(Entity):
    """
    An ARES intent
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/ares_config"
        }
    )

    type: str = Field(
        default=...,
        description="""String describing the python type""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "type",
                "domain_of": ["ARESGoal", "ARESStrategy", "AresEvaluator", "Connector"],
            }
        },
    )
    seed: Optional[int] = Field(
        default=None,
        description="""Seed to be applied to model, for example, 42.""",
        json_schema_extra={
            "linkml_meta": {"alias": "seed", "domain_of": ["Connector"]}
        },
    )
    device: Optional[str] = Field(
        default="auto",
        description="""Device on which to load the model, for example, 'auto'.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "device",
                "domain_of": ["Connector"],
                "ifabsent": "auto",
            }
        },
    )
    model_configs: ModelConfig = Field(
        serialization_alias="model_config",
        default=...,
        json_schema_extra={
            "linkml_meta": {"alias": "model_configs", "domain_of": ["Connector"]}
        },
    )
    tokenizer_config: TokenizerConfig = Field(
        default=...,
        json_schema_extra={
            "linkml_meta": {"alias": "tokenizer_config", "domain_of": ["Connector"]}
        },
    )
    generate_kwargs: Optional[GenerateKwargs] = Field(
        default=None,
        json_schema_extra={
            "linkml_meta": {"alias": "generate_kwargs", "domain_of": ["Connector"]}
        },
    )
    id: str = Field(
        default=...,
        description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "id",
                "domain_of": ["Entity"],
                "slot_uri": "schema:identifier",
            }
        },
    )
    name: Optional[str] = Field(
        default=None,
        description="""A text name of this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "name",
                "domain_of": ["Entity"],
                "slot_uri": "schema:name",
            }
        },
    )
    description: Optional[str] = Field(
        default=None,
        description="""The description of an entity""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "description",
                "domain_of": ["Entity"],
                "slot_uri": "schema:description",
            }
        },
    )
    url: Optional[str] = Field(
        default=None,
        description="""An optional URL associated with this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "url",
                "domain_of": ["Entity"],
                "slot_uri": "schema:url",
            }
        },
    )
    dateCreated: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was created.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateCreated",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateCreated",
            }
        },
    )
    dateModified: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was most recently modified.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateModified",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateModified",
            }
        },
    )


class ModelConfig(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/ares_config"
        }
    )

    pretrained_model_name_or_path: str = Field(
        default=...,
        description="""pretrained_model_name_or_path""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "pretrained_model_name_or_path",
                "domain_of": ["ModelConfig", "TokenizerConfig"],
            }
        },
    )
    torch_dtype: Optional[str] = Field(
        default=None,
        description="""model_config""",
        json_schema_extra={
            "linkml_meta": {"alias": "torch_dtype", "domain_of": ["ModelConfig"]}
        },
    )


class TokenizerConfig(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/ares_config"
        }
    )

    pretrained_model_name_or_path: str = Field(
        default=...,
        description="""pretrained_model_name_or_path""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "pretrained_model_name_or_path",
                "domain_of": ["ModelConfig", "TokenizerConfig"],
            }
        },
    )
    padding_side: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "linkml_meta": {"alias": "padding_side", "domain_of": ["TokenizerConfig"]}
        },
    )


class GenerateKwargs(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/ares_config"
        }
    )

    chat_template: Optional[ChatTemplate] = Field(
        default=None,
        json_schema_extra={
            "linkml_meta": {"alias": "chat_template", "domain_of": ["GenerateKwargs"]}
        },
    )
    generate_params: Optional[GenerateParams] = Field(
        default=None,
        json_schema_extra={
            "linkml_meta": {"alias": "generate_params", "domain_of": ["GenerateKwargs"]}
        },
    )


class ChatTemplate(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/ares_config"
        }
    )

    return_tensors: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "linkml_meta": {"alias": "return_tensors", "domain_of": ["ChatTemplate"]}
        },
    )
    thinking: Optional[bool] = Field(
        default=None,
        json_schema_extra={
            "linkml_meta": {"alias": "thinking", "domain_of": ["ChatTemplate"]}
        },
    )
    return_dict: Optional[bool] = Field(
        default=None,
        json_schema_extra={
            "linkml_meta": {"alias": "return_dict", "domain_of": ["ChatTemplate"]}
        },
    )
    add_generation_prompt: Optional[bool] = Field(
        default=None,
        json_schema_extra={
            "linkml_meta": {
                "alias": "add_generation_prompt",
                "domain_of": ["ChatTemplate"],
            }
        },
    )


class GenerateParams(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/ares_config"
        }
    )

    max_new_tokens: Optional[int] = Field(
        default=None,
        json_schema_extra={
            "linkml_meta": {"alias": "max_new_tokens", "domain_of": ["GenerateParams"]}
        },
    )


class ARESConfig(Entity):
    """
    ARES uses intents to configure red-teaming.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/ares_config"
        }
    )

    target: dict[str, Connector] = Field(
        default=...,
        description="""The path to the prompts file""",
        json_schema_extra={
            "linkml_meta": {"alias": "target", "domain_of": ["ARESConfig"]}
        },
    )
    red_teaming: RedTeaming = Field(
        serialization_alias="red-teaming",
        default=...,
        description="""The path to the prompts file""",
        json_schema_extra={
            "linkml_meta": {"alias": "red_teaming", "domain_of": ["ARESConfig"]}
        },
    )
    id: str = Field(
        default=...,
        description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "id",
                "domain_of": ["Entity"],
                "slot_uri": "schema:identifier",
            }
        },
    )
    name: Optional[str] = Field(
        default=None,
        description="""A text name of this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "name",
                "domain_of": ["Entity"],
                "slot_uri": "schema:name",
            }
        },
    )
    description: Optional[str] = Field(
        default=None,
        description="""The description of an entity""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "description",
                "domain_of": ["Entity"],
                "slot_uri": "schema:description",
            }
        },
    )
    url: Optional[str] = Field(
        default=None,
        description="""An optional URL associated with this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "url",
                "domain_of": ["Entity"],
                "slot_uri": "schema:url",
            }
        },
    )
    dateCreated: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was created.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateCreated",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateCreated",
            }
        },
    )
    dateModified: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was most recently modified.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateModified",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateModified",
            }
        },
    )


class RiskToARESMapping(Entity):
    """
    ARES uses intents to configure red-teaming.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/risk_to_ares"
        }
    )

    risk_id: str = Field(
        default=...,
        description="""The path to the prompts file""",
        json_schema_extra={
            "linkml_meta": {"alias": "risk_id", "domain_of": ["RiskToARESMapping"]}
        },
    )
    ares_config: ARESConfig = Field(
        default=...,
        description="""The path to the prompts file""",
        json_schema_extra={
            "linkml_meta": {"alias": "ares_config", "domain_of": ["RiskToARESMapping"]}
        },
    )
    id: str = Field(
        default=...,
        description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "id",
                "domain_of": ["Entity"],
                "slot_uri": "schema:identifier",
            }
        },
    )
    name: Optional[str] = Field(
        default=None,
        description="""A text name of this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "name",
                "domain_of": ["Entity"],
                "slot_uri": "schema:name",
            }
        },
    )
    description: Optional[str] = Field(
        default=None,
        description="""The description of an entity""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "description",
                "domain_of": ["Entity"],
                "slot_uri": "schema:description",
            }
        },
    )
    url: Optional[str] = Field(
        default=None,
        description="""An optional URL associated with this instance.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "url",
                "domain_of": ["Entity"],
                "slot_uri": "schema:url",
            }
        },
    )
    dateCreated: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was created.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateCreated",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateCreated",
            }
        },
    )
    dateModified: Optional[date] = Field(
        default=None,
        description="""The date on which the entity was most recently modified.""",
        json_schema_extra={
            "linkml_meta": {
                "alias": "dateModified",
                "domain_of": ["Entity"],
                "slot_uri": "schema:dateModified",
            }
        },
    )


class RiskToARES(ConfiguredBaseModel):
    """
    An umbrella object that holds the ontology class instances
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {
            "from_schema": "https://ibm.github.io/ran-ares-integration/ontology/risk_to_ares",
            "tree_root": True,
        }
    )

    mappings: Optional[list[RiskToARESMapping]] = Field(
        default=None,
        description="""A list of ares goals""",
        json_schema_extra={
            "linkml_meta": {"alias": "mappings", "domain_of": ["RiskToARES"]}
        },
    )


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
Entity.model_rebuild()
ARESGoal.model_rebuild()
ARESStrategy.model_rebuild()
AresEvaluator.model_rebuild()
AresIntent.model_rebuild()
RedTeaming.model_rebuild()
Connector.model_rebuild()
ModelConfig.model_rebuild()
TokenizerConfig.model_rebuild()
GenerateKwargs.model_rebuild()
ChatTemplate.model_rebuild()
GenerateParams.model_rebuild()
ARESConfig.model_rebuild()
RiskToARESMapping.model_rebuild()
RiskToARES.model_rebuild()
