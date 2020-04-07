# Issue: Failed to run asynchronous fl using NodeClient
## Reproduce problem
### Build Environment
```
$ mkdir test_async
$ cd test_async/
$ git clone https://github.com/ntuyoyo0/PyGrid.git
$ cd PyGrid/
$ git checkout yoyo0-dev
$ pip install -r requirements.txt
(choose 'wipe' option)
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

#### Method 3: sychronous fl (use fit() instead of async_fit())

```
$ python start_nodeClients.py
$ python run_sync_privategrid_client.py
```

#### Method 4: aychronous fl with WebsocketClientWorker 
```
$ python start_websocket_servers.py
$ python run_async_websocket_client.py
```
## What I found
It would be work if I using WebsocketClientWorker. Also, It works if I ran worker.fit() using NodeClient.
Based on the error messages in the server side, you'll find that server cannot find the obj with "0" key.
The obj with "0" key is the loss value returned by fit() in federated_client.py
I found that it wouldn't run fit() in federated_client.py when i ran worker.async_fit()

I don't know how to fix it :(

## Snapshot
### Error messages in the server side
![image](https://github.com/ntuyoyo0/PyGrid/blob/yoyo0-dev/issue_code_0407/server.png)
### Error messages in the client side
![image](https://github.com/ntuyoyo0/PyGrid/blob/yoyo0-dev/issue_code_0407/client.png)