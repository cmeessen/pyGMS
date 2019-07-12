help:
	@echo "Commands:"
	@echo ""
	@echo "    pycodestyle          check for code style conventions"
	@echo "    pep8                 check for pep8 conformity"
	@echo "    conda                install pyGMS with anaconda"
	@echo "    test                 execute tests"
	@echo "    coverage             update coverage metrics on codacy.com"
	@echo ""

pycodestyle:
	pycodestyle --show-source --ignore=W503,E226,E241,D213 pygms/*

pep8:
	pycodestyle --show-source --show-pep8 --ignore=W503,E226,E241,D213 pygms/*

conda:
	conda env create -f environment.yml

test:
	python tests/test.py

coverage:
	coverage run tests/test.py
	coverage xml
	python-codacy-coverage -r coverage.xml
