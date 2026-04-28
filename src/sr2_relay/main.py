import litellm
from litellm.proxy.proxy_server import app

from sr2_relay.api import router
from sr2_relay.sr2_handler import SR2Handler


@app.on_event("startup")
async def enable_pass_through():
  if proxy_module.llm_router:
    proxy_module.llm_router.router_general_settings.pass_through_all_models = True


app.include_router(router, prefix="/sr2")

litellm.logging_callback_manager.add_litellm_callback(SR2Handler())
