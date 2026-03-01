import importlib


def test_src_main_module_exposes_train_main():
    mod = importlib.import_module("src.__main__")
    assert hasattr(mod, "train_main")
    assert callable(mod.train_main)
