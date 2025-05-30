from lit_saint.config import SaintConfig
from omegaconf import OmegaConf

# Create a default SaintConfig object
conf = SaintConfig()

# Convert to OmegaConf DictConfig (optional, but shown in lit-saint docs for saving)
dict_conf = OmegaConf.create(conf)

# Print the OmegaConf string representation
print("--- Default SaintConfig YAML representation ---")
print(OmegaConf.to_yaml(dict_conf))

# Specifically check for cutmix related attributes
print("\n--- CutMixConfig attributes ---")
if hasattr(conf, 'cutmix') and conf.cutmix is not None:
    print(f"cutmix type: {type(conf.cutmix)}")
    print(f"cutmix attributes: {conf.cutmix.__dict__}")
    if hasattr(conf.cutmix, 'type'):
        print(f"cutmix.type: {conf.cutmix.type}")
else:
    print("No 'cutmix' attribute or it is None.")

print("\n--- MixupConfig attributes ---")
if hasattr(conf, 'mixup') and conf.mixup is not None:
    print(f"mixup type: {type(conf.mixup)}")
    print(f"mixup attributes: {conf.mixup.__dict__}")
    if hasattr(conf.mixup, 'type'):
        print(f"mixup.type: {conf.mixup.type}")
else:
    print("No 'mixup' attribute or it is None.") 