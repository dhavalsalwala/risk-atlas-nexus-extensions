import os
from pathlib import Path

import yaml
from linkml_runtime.loaders import yaml_loader

from ran_ares_integration.datamodel.risk_to_ares_ontology import RiskToARES


ASSETS_DIR_PATH = Path(__file__).parent.absolute()

ARES_TARGETS = yaml.safe_load(
    Path(os.path.join(ASSETS_DIR_PATH, "connectors.yaml")).read_text()
)

RISK_TO_ARES_MAPPINGS: RiskToARES = yaml_loader.load_any(
    source=yaml_loader.load_as_dict(
        source=os.path.join(
            ASSETS_DIR_PATH, "knowledge_graph", "risk_to_ares_mappings.yaml"
        )
    ),
    target_class=RiskToARES,
)
