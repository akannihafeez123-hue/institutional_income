```bash
# Build image
docker build -t legendary-empire .

# Run container
docker run -d \
  --name legendary-empire \
  -p 8080:8080 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  legendary-empire
```
