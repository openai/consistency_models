from setuptools import setup

setup(
    name="consistency-models",
    py_modules=["cm", "evaluations"],
    install_requires=[
        "blobfile>=1.0.5",
        "torch",
        "tqdm",
        "numpy",
        "scipy",
        "pandas",
        "Cython",
        "piq==0.7.0",
        "joblib==0.14.0",
        "albumentations==0.4.3",
        "lmdb",
        "clip @ git+https://github.com/openai/CLIP.git",
        "mpi4py",
        "flash-attn==0.2.8",
        "pillow",
    ],
)
