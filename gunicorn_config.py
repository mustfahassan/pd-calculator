import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes - reduced for TensorFlow compatibility
workers = 1  # Using single worker to avoid TF thread conflicts
threads = 4  # Using threads instead of multiple workers
worker_class = 'gthread'  # Using threads
worker_connections = 1000
timeout = 60  # Increased timeout for TF operations
keepalive = 2

# Logging
accesslog = 'logs/gunicorn-access.log'
errorlog = 'logs/gunicorn-error.log'
loglevel = 'debug'  # Set to debug for more information

# Process naming
proc_name = 'pd_measurement'

# Server mechanics
daemon = False
pidfile = 'logs/gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# Preload app to avoid multiple TF initializations
preload_app = True

# SSL
keyfile = None
certfile = None

def post_fork(server, worker):
    # Ensure TensorFlow is initialized after fork
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)