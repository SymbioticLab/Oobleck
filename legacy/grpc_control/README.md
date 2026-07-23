[https://grpc.io/docs/languages/python/basics/](https://grpc.io/docs/languages/python/basics/)

# Rebuild gRPC interface

Run on the current directory:

```bash
$ pip install grpcio-tools
$ python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. ./master_service.proto
```