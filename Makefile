service=time-series-forecast
package = github.com/allenwind/${service}
version = $(shell git describe --long --tags --dirty | awk '{print substr($$1,2)}')

.PHONY: pyclean
pyclean:
    find . -type d -name "__pycache__" | xargs rm -rf
