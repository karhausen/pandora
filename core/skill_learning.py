from __future__ import annotations
import json, re, uuid
from collections import Counter
from datetime import datetime, UTC
from .config import PROPOSALS_DIR
from .episodic_memory import EpisodicMemory
from .models import SkillMeta, SkillProposal
class SkillLearningEngine:
    def __init__(self, episodic_memory: EpisodicMemory | None = None):
        self.episodic_memory=episodic_memory or EpisodicMemory()
    def _safe_id(self,text): return re.sub(r"[^a-z0-9_]+","_",text.lower()).strip("_")
    def find_repeated_tool_sequences(self,min_count:int=2)->list[dict]:
        counter=Counter(tuple(seq) for seq in self.episodic_memory.successful_tool_sequences())
        return [{"sequence":list(seq),"count":count} for seq,count in counter.items() if len(seq)>=2 and count>=min_count]
    def propose_skills_from_patterns(self,min_count:int=2)->list[dict]:
        proposals=[]
        for pat in self.find_repeated_tool_sequences(min_count):
            seq=pat["sequence"]; skill_id=self._safe_id("skill_"+"_then_".join(seq)); steps=[]; prev=None
            for idx,tool_id in enumerate(seq):
                save=f"{tool_id}_{idx+1}"; steps.append({"id":f"step_{idx+1}_{tool_id}","type":"tool","tool_id":tool_id,"input_map":{"text":"input.text" if idx==0 else f"context.{prev}.text"},"save_as":save}); prev=save
            skill=SkillMeta.model_validate({"id":skill_id,"name":"Learned "+" Then ".join(seq),"description":f"Learned workflow from repeated successful sequence: {' -> '.join(seq)}.","version":"0.1.0","status":"GENERATED","security_level":"SAFE","required_tools":list(dict.fromkeys(seq)),"input_schema":{"text":"str"},"output_schema":{"context":"dict"},"steps":steps})
            proposal=SkillProposal(id=str(uuid.uuid4()),name=f"Proposal for {skill_id}",description=skill.description,reason=f"Sequence {' -> '.join(seq)} occurred successfully {pat['count']} times.",skill=skill,evidence=pat)
            proposals.append(self._save(proposal))
        return proposals
    def _save(self, proposal: SkillProposal)->dict:
        d=PROPOSALS_DIR/"skills"/proposal.skill.id; d.mkdir(parents=True,exist_ok=True); payload=proposal.model_dump(mode="json"); payload["created_at"]=datetime.now(UTC).isoformat()
        (d/"learned_skill_proposal.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8"); (d/f"{proposal.skill.id}.json").write_text(json.dumps(proposal.skill.model_dump(mode="json"),indent=2,ensure_ascii=False),encoding="utf-8")
        return {"proposal_id":proposal.id,"skill_id":proposal.skill.id,"proposal_dir":str(d),"reason":proposal.reason,"status":proposal.status}
