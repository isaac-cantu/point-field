"""Fallback executor where Jupyter sockets are unavailable.
Runs actual notebook cells in one IPython process, captures text/figures.
Not a substitute for final Restart Kernel / Run All on the user's machine.
"""
import io
import sys
from pathlib import Path
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.display import display, Image

def show(*args,**kwargs):
    for num in plt.get_fignums():
        fig=plt.figure(num);buf=io.BytesIO();fig.savefig(buf,format='png',bbox_inches='tight')
        display(Image(data=buf.getvalue()))
    plt.close('all')
plt.show=show
p=Path(sys.argv[1]);out=Path(sys.argv[2])
nb=nbformat.read(p,as_version=4);shell=InteractiveShell.instance();count=0
for i,c in enumerate(nb.cells):
    if c.cell_type!='code':continue
    count+=1;print('Executing cell',i,flush=True)
    with capture_output(display=True) as cap:
        result=shell.run_cell(c.source,store_history=True)
    c.execution_count=count;c.outputs=[]
    if cap.stdout:c.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=cap.stdout))
    if cap.stderr:c.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=cap.stderr))
    for o in cap.outputs:c.outputs.append(nbformat.v4.new_output('display_data',data=o.data,metadata=o.metadata))
    nb.metadata['execution_method']='IPython in-process; TCP/IPC kernel sockets unavailable in execution environment'
    nbformat.write(nb,out)
    if result.error_before_exec or result.error_in_exec:raise RuntimeError(f'Cell {i} failed: {result.error_before_exec or result.error_in_exec}')
print('All cells passed',flush=True)
