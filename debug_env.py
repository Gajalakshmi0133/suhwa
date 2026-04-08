import sys
import os
try:
    import flask_socketio
    print("flask_socketio imported successfully")
except ImportError:
    print("flask_socketio NOT found")

print("\nSYS PATH:")
for p in sys.path:
    print(p)

print("\nSITE-PACKAGES CONTENT (partial):")
site_pkgs = [p for p in sys.path if 'site-packages' in p]
if site_pkgs:
    try:
        print(os.listdir(site_pkgs[0])[:20])
    except:
        print("Could not list site-packages")
