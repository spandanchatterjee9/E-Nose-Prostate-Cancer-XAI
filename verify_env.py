import sys
import os
import subprocess
import socket

def print_result(check_name, success, message=""):
    status = "[ OK ]" if success else "[FAIL]"
    print(f"{status} {check_name:<30} {message}")
    return success

def check_port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False

def main():
    print("=" * 60)
    print("          E-Nose Environment Verification Utility          ")
    print("=" * 60)
    
    all_ok = True
    
    # 1. Python Check
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    is_py_ok = sys.version_info >= (3, 9)
    all_ok &= print_result("Python Version (>= 3.9)", is_py_ok, f"Found: {py_ver}")
    
    # 2. Virtual Env Check
    is_venv = sys.prefix != sys.base_prefix or 'VIRTUAL_ENV' in os.environ
    all_ok &= print_result("Virtual Environment Active", is_venv, f"Prefix: {sys.prefix}")
    
    # 3. Node.js Check
    node_found = False
    node_ver = "Not found"
    try:
        res = subprocess.run(["node", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        if res.returncode == 0:
            node_found = True
            node_ver = res.stdout.strip()
    except Exception:
        pass
    all_ok &= print_result("Node.js Installed", node_found, f"Found: {node_ver}")
    
    # 4. npm Check
    npm_found = False
    npm_ver = "Not found"
    try:
        res = subprocess.run(["npm", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        if res.returncode == 0:
            npm_found = True
            npm_ver = res.stdout.strip()
    except Exception:
        pass
    all_ok &= print_result("npm Installed", npm_found, f"Found: {npm_ver}")
    
    # 5. Libraries check
    libs = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("sqlalchemy", "SQLAlchemy"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-Learn"),
        ("xgboost", "XGBoost"),
        ("tensorflow", "TensorFlow"),
        ("shap", "SHAP")
    ]
    
    print("\nVerifying Python Packages:")
    for lib_name, print_name in libs:
        try:
            mod = __import__(lib_name)
            ver = getattr(mod, "__version__", "Importable")
            # Special handling for scikit-learn
            if lib_name == "sklearn":
                import sklearn
                ver = sklearn.__version__
            print_result(f"  {print_name}", True, f"Version: {ver}")
        except ImportError:
            print_result(f"  {print_name}", False, "Not installed")
            all_ok = False
            
    # 6. TensorFlow CUDA/GPU check
    tf_gpu = False
    gpu_devices = []
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        tf_gpu = len(gpus) > 0
        gpu_devices = [g.name for g in gpus]
    except Exception:
        pass
    print_result("TensorFlow CUDA / GPU Support", tf_gpu, f"Devices: {gpu_devices}" if tf_gpu else "Running on CPU (CUDA not detected or not configured)")
    
    # 7. Ports check
    print("\nVerifying Server Ports:")
    port_8000_free = check_port_free(8000)
    print_result("  Backend Port 8000 (FastAPI)", port_8000_free, "Free" if port_8000_free else "PORT CONFLICT - In use by another service!")
    
    port_3000_free = check_port_free(3000)
    print_result("  Frontend Port 3000 (Next.js)", port_3000_free, "Free" if port_3000_free else "PORT CONFLICT - In use by another service!")
    
    print("=" * 60)
    if all_ok:
        print(" SUCCESS: Environment is healthy and fully configured! ")
    else:
        print(" WARNING: Some requirements are missing or ports are occupied. ")
        print(" Refer to setup instructions or troubleshooting documentation. ")
    print("=" * 60)
    
    # Return exit code based on crucial requirements (python, node, npm, libraries)
    essential_ok = is_py_ok and is_venv and node_found and npm_found
    for lib_name, _ in libs:
        try:
            __import__(lib_name)
        except ImportError:
            essential_ok = False
            
    # Exit with 0 if essential setup is fine, 1 if something crucial is missing
    sys.exit(0 if essential_ok else 1)

if __name__ == '__main__':
    main()
