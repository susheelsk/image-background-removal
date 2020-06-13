from main import process
import os
import platform
import subprocess
import sys
import threading
import webview

def worker_thread(window, inputfiles, model):
    print('processing started ...')
    count = 0
    for inputfile in inputfiles:
      count += 1
      (inputfilepath, inputfilename) = os.path.split(inputfile)      
      outputfile = os.path.join(
        inputfilepath,
        'bg-removed',
        # only support PNG files as of now
        os.path.splitext(inputfilename)[0] + '.png'
      )
      window.evaluate_js(
        "window.app.fileUploadButton.textContent = 'Processing "
          + str(count) + ' of ' + str(len(inputfiles)) + " ...'"
      )
      process(inputfile, outputfile, model)
    window.evaluate_js("window.app.fileUploadButton.textContent = 'Select photos'")
    print('processing complete')
    open_folder(os.path.join(inputfilepath, 'bg-removed'))

def onWindowStart(window):
  def openFileDialog():
      file_types = ('Image Files (*.png;*.jpg;*.jpeg)', 'All files (*.*)')

      inputfiles = window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True, file_types=file_types)
      print(inputfiles)

      model = window.evaluate_js("window.app.getModel()")
      print('use model =', model)

      if inputfiles != None:
        window.evaluate_js("window.app.fileUploadButton.disabled = true")
        workerThread = threading.Thread(target=worker_thread, args=(window, inputfiles, model,))
        workerThread.start()
        workerThread.join()
        window.evaluate_js("window.app.fileUploadButton.disabled = false")

  # expose a function during the runtime
  window.expose(openFileDialog)

def open_folder(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

def getfile(filename):
  dir = os.path.dirname(__file__)
  path = os.path.join(dir, filename)
  f = open(path, 'r')
  content = f.read()
  f.close()
  return content

html = getfile('index.html')
window = webview.create_window('Automated BG Removal Tool', html=html)
webview.start(onWindowStart, window)