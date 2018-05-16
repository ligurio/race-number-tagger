### Image classification demo running as a Flask web server.

## Requirements

The demo server requires Python with some dependencies. To make sure you have
the dependencies, please run `pip install -r requirements.txt`.

## Run

Running `python examples/web_demo/app.py` will bring up the demo server,
accessible at `http://0.0.0.0:5000`. You can enable debug mode of the web
server, or switch to a different port:

    % python app.py -h
    Usage: app.py [options]

    Options:
      -h, --help            show this help message and exit
      -d, --debug           enable debug mode
      -p PORT, --port=PORT  which port to serve content on
