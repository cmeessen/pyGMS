help:
	@echo "Commands:"
	@echo ""
	@echo "    pycodestyle          check for code style conventions"
	@echo ""

pycodestyle:
	pycodestyle --show-source --ignore=W503,E226,E241,D213 pygms/*

pep8:
	pycodestyle --show-source --show-pep8 --ignore=W503,E226,E241,D213 pygms/*

conda:
	conda env create -f environment.yml
