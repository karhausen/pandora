from pathlib import Path
import os
class ObsidianVault:
    def __init__(self):
        self.vault_path=Path(os.getenv("OBSIDIAN_VAULT_PATH",""))
        self.inbox=os.getenv("OBSIDIAN_INBOX_DIR","Pandora_Inbox")
    def status(self):
        return {"enabled":bool(os.getenv("OBSIDIAN_VAULT_ENABLED","false").lower()=="true"),
                "vault_exists":self.vault_path.exists(),
                "vault_path":str(self.vault_path),
                "inbox":self.inbox}
