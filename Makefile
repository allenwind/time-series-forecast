service=time-series-forecast
package = github.com/allenwind/${service}
version = $(shell git describe --long --tags --dirty | awk '{print substr($$1,2)}')


.PHONY: pyclean
pyclean:
    find . -type d -name "__pycache__" | xargs rm -rf


.PHONY: python3
docker-python3:
    rm -rf python3
    sudo docker run --rm -v $(shell pwd):/heptapod conda/miniconda3 \
    conda create -y --prefix /heptapod/python3 --file /heptapod/requirements.txt --copy --no-default-packages python=3.6
