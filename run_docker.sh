#!/bin/bash

docker run -it -v $(pwd):/app/main --gpus all gnntabledetection bash
cd main