from setuptools import setup, find_packages

setup(
    name="ai_agents",
    version="0.1.0",
    description="Data harmonization demo with embeddings and LangChain",
    author="Abheesht Roy",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas","numpy","langchain","sentence-transformers",
        "openai","faiss-cpu","python-dotenv",
    ],
)
