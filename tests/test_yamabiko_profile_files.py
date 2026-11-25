"""Every version in models/ has the files its profile names (a step folder copied without its runtimes broke step7c)."""
import os

from flute_rl.yamabiko.profiles import load_profiles


def test_every_profile_names_files_that_exist():
    missing = {name: [v for k, v in p.items() if k.endswith("_path") and not os.path.exists(v)]
               for name, p in load_profiles().items()}
    assert not {n: m for n, m in missing.items() if m}
