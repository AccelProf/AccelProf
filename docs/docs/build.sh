# # dependencies
# pip install -r docs/requirements.txt
# pip install sphinx-autobuild

# build
sphinx-build -b html source/ _build/html
sphinx-autobuild source/ _build/html