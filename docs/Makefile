# Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXGEN     ?= sphinx-apidoc
NBCONVERT     ?= jupyter
PYTHON        ?= python
PIP           ?= pip
SOURCEDIR     = .
BUILDDIR      = _build
TEMPLATESDIR  = templates
STATICDIR     = _static

example_notebooks := $(wildcard ../examples/*.ipynb)
example_names = $(foreach path, $(example_notebooks), $(basename $(notdir $(path))))
example_nbs = $(foreach name, $(example_names), examples/$(name).ipynb)

html: conf.py index.rst install.rst examples.rst README.rst xmcdpy.rst  $(example_nbs) .deps
	echo $(example_nbs)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	touch "$(BUILDDIR)/html/.nojekyll"
	rm -rvf "$(BUILDDIR)/html/_static/examples/*"
	# mv "$(BUILDDIR)/html/_static/examples" "$(BUILDDIR)/html/examples" || true
	# mv "$(BUILDDIR)/html/_static/images" "$(BUILDDIR)/html/images" || true
	# cp "$(STATICDIR)/favicon.ico" "$(BUILDDIR)/html/favicon.ico" || true

.deps: requirements.txt
	$(PIP) install -r requirements.txt
	$(PIP) freeze > .deps

README.rst: ../README.md .deps
	pandoc --from=gfm --to=rst --output=README.rst ../README.md


examples/%.ipynb: ../examples/%.ipynb .deps
	python -c "import os; os.makedirs('./examples', exist_ok=True)"
	cp $< $(abspath $@) 

# $(STATICDIR)/examples/%.html: ../examples/%.ipynb .deps
# 	python -c "import os; os.makedirs('_static/examples', exist_ok=True)"
# 	$(NBCONVERT) nbconvert --to html --output $(abspath $@) $<

# examples.%.rst: $(STATICDIR)/examples/%.html examples.template
# 	$(PYTHON) -c "open('$@', 'w').write(open('examples.template').read().format(name='$(*F)', url='$<'))"

xmcdpy.rst: .deps
	$(SPHINXGEN) -eTf -t "$(TEMPLATESDIR)" -o "$(SOURCEDIR)" ../xmcdpy

clean:
	rm -rvf "$(BUILDDIR)" "$(STATICDIR)/examples" examples.*.rst xmcdpy*.rst .deps README.rst

# # All other Sphinx builders
#%: conf.py index.rst install.rst examples.rst README.rst trimesh.rst $(example_rsts) $(example_htmls)
#	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
#
#.PHONY: %