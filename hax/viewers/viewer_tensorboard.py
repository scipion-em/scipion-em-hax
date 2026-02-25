# **************************************************************************
# *
# * Authors:  David Herreros (dherreros@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


import os
from glob import glob
import time
import subprocess
import multiprocessing as mp
import psutil
import socket
import webbrowser
from contextlib import closing
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import urllib.request

from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.gui.dialog import showError

from pwem.viewers import DataViewer

from hax.protocols import (JaxProtFlexibleAlignmentHetSiren, JaxProtTrainFlexConsensus, JaxProtAngularAlignmentReconSiren,
                           JaxProtImageAdjustment, JaxProtVolumeAdjustment, JaxProtReconstructMoDART)

import hax


def get_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def wait_for_server_ready(url, timeout=30):
    """
    Polls the URL until it returns a 200 OK, ensuring the app is actually loaded.
    """
    print(f"Waiting for TensorBoard to load at {url}...", end="", flush=True)
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # We set a short timeout for the request itself so we don't hang
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    print(" Ready!")
                    return True
        except (urllib.error.URLError, socket.timeout, ConnectionResetError):
            # Server not ready yet, wait and retry
            time.sleep(0.5)
            print(".", end="", flush=True)

    print("\nTimed out waiting for TensorBoard.")
    return False


def launch_smart_tensorboard(logdir_path):
    # 1. Setup Ports
    tb_port = get_free_port()
    wrapper_port = get_free_port()

    # 2. Launch TensorBoard Process
    # Ensure 'hax' is imported or replace with "tensorboard" string
    try:
        program = hax.Plugin.getProgram("tensorboard", gpu=None, uses_project_manager=False)
    except NameError:
        program = "tensorboard"

        # Note: --bind_all allows the internal iframe request to work reliably
    args = f" --logdir {logdir_path} --port {tb_port} --bind_all"

    # Start the process
    tb_process = subprocess.Popen(f"{program} {args}", shell=True)

    # 3. BLOCK HERE until TensorBoard is actually ready
    tb_url = f"http://localhost:{tb_port}/"
    if not wait_for_server_ready(tb_url):
        print("Failed to start TensorBoard. Exiting.")
        tb_process.terminate()
        return

    # 4. Define the Heartbeat Wrapper (Auto-Close Logic)
    class HeartbeatHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return  # Silence logs

        def do_GET(self):
            if self.path == '/heartbeat':
                self.server.last_beat = time.time()
                self.send_response(200)
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                # We embed the NOW-READY TensorBoard URL
                html = f"""
                <html>
                <head>
                    <title>TensorBoard</title>
                    <style>body,html,iframe{{width:100%;height:100%;margin:0;border:none;overflow:hidden;}}</style>
                    <script>
                        setInterval(() => fetch('/heartbeat').catch(console.error), 2000);
                        window.onbeforeunload = () => fetch('/heartbeat?closing=true');
                    </script>
                </head>
                <body>
                    <iframe src="{tb_url}"></iframe>
                </body>
                </html>
                """
                self.wfile.write(html.encode('utf-8'))

    # 5. Start the Wrapper Server
    server = HTTPServer(('localhost', wrapper_port), HeartbeatHandler)
    server.last_beat = time.time()

    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()

    # 6. Open Browser (Only now!)
    final_url = f"http://localhost:{wrapper_port}/"
    print(f"Opening browser at {final_url}")
    webbrowser.open(final_url)

    # 7. Monitor Loop
    try:
        while True:
            time.sleep(1)
            # Kill if no heartbeat for 5 seconds OR if TB process crashes
            if time.time() - server.last_beat > 5:
                print("\nBrowser tab closed. Stopping TensorBoard...")
                break
            if tb_process.poll() is not None:
                print("\nTensorBoard process died. Exiting...")
                break
    except KeyboardInterrupt:
        pass

    # 8. Cleanup
    try:
        parent = psutil.Process(tb_process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        tb_process.terminate()
    except:
        pass
    server.shutdown()


class JaxTensorboardViewer(DataViewer):
    """ Tensorboard visualization of neural networks """
    _label = 'viewer Tensorboard'
    _targets = [JaxProtFlexibleAlignmentHetSiren,
                JaxProtTrainFlexConsensus,
                JaxProtAngularAlignmentReconSiren,
                JaxProtImageAdjustment,
                JaxProtVolumeAdjustment,
                JaxProtReconstructMoDART]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        DataViewer.__init__(self, **kwargs)
        self._data = None

    def _visualize(self, obj, **kwargs):
        logdir_path = glob(self.protocol._getExtraPath("*_metrics/"))[0]

        if not os.path.isdir(logdir_path):
            if hasattr(self.protocol, "tensorboard"):
                if not self.protocol.tensorboard.get():
                    msg = ("Tensorboard cannot be opened because the option \"Allow Tensorboard visualization\" "
                           "in the protocol form has been set to \"No\". If you want to visualize one of the "
                           "outputs generated by this protocol, please, right click on the deried output from the "
                           "output list and choose the desired viewer to open it.")
            else:
                msg = "Tensorboard log files have not been properly generated."
            showError(title="Tensorboard log files not found", msg=msg, parent=self.getTkRoot())
            return []

        p = mp.Process(target=launch_smart_tensorboard, args=(logdir_path,), daemon=True)
        p.start()

        return []
