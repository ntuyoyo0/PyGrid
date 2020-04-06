# Issue: Failed to run asynchronous fl using NodeClient
## Reproduce problem
### Build Environment
```
$ mkdir test_async
$ cd test_async/
$ git clone https://github.com/ntuyoyo0/PyGrid.git
$ cd PyGrid/
$ git checkout yoyo0-dev
(choose 'wipe' option)
$ pip install -r requirements.txt
```

```
$ cd issue_code_0407/
$ cp websocket_app2.py ../app/websocket/
$ mv ../app/websocket/app/__init__.py ../app/websocket/app/__init__backup.py
$ cp __init__.py ../app/websocket/app/
```

### Run the code 
#### Method 1: run workers at once (one command with one terminal window)

```
$ python start_nodeClients.py
$ python run_async_privategrid_client.py
```

#### Method 2: run workers one by one (one command with one terminal window)
```
$ python ../app/websocket/websocket_app2.py --host 0.0.0.0 --port 6666 --id alice --notebook mnist-parallel
$ python ../app/websocket/websocket_app2.py --host 0.0.0.0 --port 6667 --id bob --notebook mnist-parallel
$ python ../app/websocket/websocket_app2.py --host 0.0.0.0 --port 6668 --id charlie --notebook mnist-parallel
$ python ../app/websocket/websocket_app2.py --host 0.0.0.0 --port 6669 --id testing --notebook mnist-parallel
$ python run_async_privategrid_client.py
```
