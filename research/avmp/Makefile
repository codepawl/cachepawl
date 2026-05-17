# AVMP paper build targets.
#
# Run all commands from the research/avmp/ directory:
#
#   make figures   # regenerate the four figures
#   make tables    # regenerate the five tables
#   make paper     # build paper.pdf (depends on figures, tables, and a
#                  # one-time pass over acmart.ins to extract acmart.cls
#                  # from the vendored .dtx source)
#   make all       # same as make paper
#   make clean     # remove generated figures, tables, and LaTeX
#                  # build artifacts
#   make arxiv     # produce arxiv-submission.tar.gz with .bbl,
#                  # acmart sources, sections, generated figures and
#                  # tables, ready for arXiv upload

.PHONY: figures tables paper all clean arxiv class

PAPER := paper

figures:
	uv run --project .. python scripts/generate_figures.py figures/generated

tables:
	uv run --project .. python scripts/generate_tables.py tables/generated

# Extract acmart.cls (and the helper .sty files that acmart.ins emits)
# from the vendored acmart.dtx source on first build.
acmart.cls:
	latex acmart.ins

class: acmart.cls

paper: figures tables class
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

all: paper

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz *.fdb_latexmk *.fls
	rm -f $(PAPER).pdf acmart.cls acmart.cfg
	rm -rf figures/generated tables/generated arxiv-submission arxiv-submission.tar.gz

arxiv: paper
	mkdir -p arxiv-submission/sections arxiv-submission/bibliography
	mkdir -p arxiv-submission/figures arxiv-submission/tables
	cp $(PAPER).tex $(PAPER).bbl arxiv-submission/
	cp acmart.cls ACM-Reference-Format.bst arxiv-submission/
	cp sections/*.tex arxiv-submission/sections/
	cp bibliography/refs.bib arxiv-submission/bibliography/
	cp -r figures/static arxiv-submission/figures/
	cp figures/generated/*.pdf arxiv-submission/figures/
	cp tables/generated/*.tex arxiv-submission/tables/
	tar -czf arxiv-submission.tar.gz arxiv-submission
	rm -rf arxiv-submission
	@echo "arxiv-submission.tar.gz ready for upload"
