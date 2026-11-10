import csv, math
LABELS=['winner_model_a','winner_model_b','winner_tie']
def read(p):
    with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def score(truth_path,pred_path):
    truth=read(truth_path); pred={r['id']:r for r in read(pred_path)}; total=0.0
    for row in truth:
        q=pred[row['id']]; ps=[float(q[k]) for k in LABELS]
        if any(not math.isfinite(x) or x<0 or x>1 for x in ps) or abs(sum(ps)-1)>1e-6: raise ValueError('invalid probabilities')
        y=next(k for k in LABELS if row[k]=='1'); total-=math.log(max(1e-15,min(1-1e-15,float(q[y]))))
    return total/len(truth)
