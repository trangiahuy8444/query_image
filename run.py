import subprocess
import time
import os
import sys
from threading import Thread
from visualization import QueryMetricsVisualizer

def run_flask_server():
    """Run Flask server in a separate process"""
    print("Starting Flask server...")
    subprocess.run([sys.executable, "app.py"])

def run_visualization():
    """Run visualization after server is ready"""
    # Wait for server to start
    time.sleep(5)
    
    # Initialize visualizer and process the image
    image_path = "image_test/150542.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
        
    visualizer = QueryMetricsVisualizer()
    visualizer.query_and_visualize(image_path)

def main():
    # Start Flask server in a separate thread
    server_thread = Thread(target=run_flask_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Run visualization
    run_visualization()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

if __name__ == '__main__':
    main() 