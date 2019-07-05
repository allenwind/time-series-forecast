service=time-series-forecast
package = github.smartx.com/smartx/${service}
version = $(shell git describe --long --tags --dirty | awk '{print substr($$1,2)}')
build_dir = _build

.PHONY: clean
clean:
	rm *.png
	rm *.hdf5

.PHONY: pyclean
pyclean:
	find . -type d -name "__pycache__" | xargs rm -rf

.PHONY: utests
utests:
	python3 utests.py

.PHONY: update
update:
	git clone git@github.com:allenwind/time-series-features.git
	cp -r time-series-features/tsfeatures tsforecast
	rm -rf time-series-features

.PHONY: docker-test
docker-test:

.PHONY: mlp
mlp:
	mkdir -p weights
	touch weights/dummy
	rm weights/*
	python3 train_mlp.py

.PHONY: ensemble
ensemble:
	python3 train_ensemble.py

.PHONY: linear
linear:
	python3 train_linear.py
