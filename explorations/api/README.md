# Setup
This assumes that embeddings are created and stored in a vector data store. MongoDB Atlas Vector Search is used in this prototype. Check out the loader notebook to create embeddings.

## Steps
1. set up gcp credentials
1. make sure dependencies are installed
1. `fastapi dev main.py`

# Usage
FastAPI will by default run on `localhost:8000`. The primary api route is `POST /api/query` which accepts an argument of the following schema:

```json
{
    "text": "under what circumstances can i have more than one m0 in a project",
    "context": false
}
```

In a scenario where you request `"context": true` you will receive a long context object. Otherwise you will receive an object without it, with the following as an example:
```json
{
    "answer":"You can typically only deploy one M0 cluster per project. However, if you have a sales contract with MongoDB, you might be able to deploy more. Contact a MongoDB sales representative to learn more.\n"
}
```