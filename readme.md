### Deploy Command


```
gunicorn -w 1 -b :8080 --timeout 300  --daemon   --access-logfile /var/log/gunicorn/access.log   --error-logfile /var/log/gunicorn/error.log   --capture-output   --log-level debug   flux:app
```