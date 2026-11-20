"""Check Central for pending complaints and run the graph for each one."""

import logging
from time import sleep

from agent.graph import run_triage
from utilities.mailpit import pending_messages

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    logger.info("Watching Central for complaints.")
    while True:
        try:
            for message in pending_messages():
                try:
                    result = run_triage({"rfc_message_id": message["MessageID"]})
                    logger.info("Routed %s to Manual Review (%s).", message["ID"], result["forwarded_id"])
                except Exception:
                    logger.exception("Could not route %s. Will retry on the next check.", message["ID"])
        except Exception:
            logger.exception("Could not check Mailpit. Will retry on the next check.")
        sleep(2)


if __name__ == "__main__":
    main()
