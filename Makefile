# Build entry points. `make help` lists them.
#
# Every target is seeded and config-free on purpose. The main-text numbers come
# from three scripts, each writing a JSON under experiments/; the two figures
# read back reparam_invariance_results.json and containment_audit_results_v2.json.

PY ?= python
TEXDIR := paper
TEX := $(TEXDIR)/main
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error

AUDIT_JSON := experiments/containment_audit_results_v2.json

TESTS := tests cio/tests irl/tests robust/tests conformal/tests

.PHONY: help paper clean test figures experiments verify

help:
	@echo "make paper       build paper/main.pdf (4-pass, runs bibtex)"
	@echo "make test        run the unit-test suite"
	@echo "make experiments rerun the three scripts behind the paper's numbers (hours)"
	@echo "make figures     redraw both figures from the recorded JSONs"
	@echo "make verify      re-check the App. C identities numerically"
	@echo "make clean       remove LaTeX byproducts (keeps main.pdf)"

paper:
	cd $(TEXDIR) && pdflatex $(LATEX_FLAGS) main.tex
	cd $(TEXDIR) && bibtex main
	cd $(TEXDIR) && pdflatex $(LATEX_FLAGS) main.tex
	cd $(TEXDIR) && pdflatex $(LATEX_FLAGS) main.tex
	@echo "built $(TEX).pdf"

# Byproducts only; paper/main.pdf is tracked and is left alone.
clean:
	rm -f $(addprefix $(TEXDIR)/main.,aux bbl blg log out toc lof lot nav snm vrb)

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. $(PY) -m pytest $(TESTS) -q

experiments:
	PYTHONPATH=. $(PY) experiments/run_reparam_invariance.py
	PYTHONPATH=. $(PY) experiments/run_containment_audit.py --out $(AUDIT_JSON)
	PYTHONPATH=. $(PY) experiments/run_center_validity.py

figures:
	cd $(TEXDIR) && PYTHONPATH=.. $(PY) make_fig1_metric.py
	cd $(TEXDIR) && PYTHONPATH=.. $(PY) make_fig2_identification_frontier.py

verify:
	cd $(TEXDIR) && PYTHONPATH=.. $(PY) verify_formulas.py
