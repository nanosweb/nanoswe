"""Reject incomplete shards and unsalvaged infrastructure failures."""
import json,sys,pathlib
out=pathlib.Path(sys.argv[1]); ids=json.load(open(sys.argv[2]))['instance_ids'];errors=[]
for iid in ids:
 for sample in range(5):
  p=out/iid/f'sample_{sample}.traj.json'
  if not p.exists():errors.append(f'missing {iid}/{sample}');continue
  d=json.loads(p.read_text());info=d.get('info',{});g=info.get('grading') or {}
  if info.get('traceback') and 'SalvagedWIP' not in info.get('exit_status',''):errors.append(f'traceback {iid}/{sample}')
  if 'resolved' not in g:errors.append(f'no grade {iid}/{sample}')
  if g.get('status')=='overlay_failed':errors.append(f'overlay_failed {iid}/{sample}')
result={'expected':5*len(ids),'errors':errors,'complete':not errors}
(out/'completion_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result));sys.exit(bool(errors))
