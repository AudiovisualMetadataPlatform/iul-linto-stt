#!/bin/env python3
import argparse
import json
import logging
import os
import time
from pathlib import Path

from stt import logger as stt_logger
from stt.processing import MODEL, USE_GPU, decode, load_wave_buffer, warmup

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger("__stt-cli__")
logger.setLevel(logging.INFO)

def main():
    logger.info("Startup...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true", help="Enable Debugging")
    parser.add_argument("--language", default="en", help="Language to use")
    parser.add_argument("file", nargs="+", help="filename or filename=output specs")
    args = parser.parse_args()
    logger_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(logger_level)
    stt_logger.setLevel(logger_level)
    logger.info(args)

    for filespec in args.file:
        if '=' in filespec:
            srcfile, dstfile = filespec.split('=', 1)
        else:
            srcfile = filespec
            dstfile = filespec + ".json"

        try:
            logger.info(f"Transcribing {srcfile} -> {dstfile}")            
            file_buffer = Path(srcfile).read_bytes()
            language = args.language
            audio_data = load_wave_buffer(file_buffer)
            
            # Transcription
            transcription = decode(audio_data, MODEL, True, language=language)
            Path(dstfile).write_text(json.dumps(transcription, ensure_ascii=False))

        except Exception as error:
            import traceback
            logger.error(traceback.format_exc())
            logger.error(repr(error))



if __name__ == "__main__":
    main()
