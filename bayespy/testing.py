# Copyright (c) 2015 Bernhard Thiel, Jaakko Luttinen
# MIT License
# From: https://github.com/Bernhard10/WarnAsError/blob/master/warnaserror.py

__author__ = 'Bernhard Thiel'

from nose.plugins import Plugin
import nose
import warnings


class WarnAsError(Plugin):

    enabled = False


    def options(self, parser, env):
        """
        Add options to command line.
        """
        super().options(parser, env)
        parser.add_option("--warn-as-error", action="store_true",
                          default=False,
                          dest="warnaserror",
                          help="Treat warnings that occur WITHIN tests as errors.")


    def configure(self, options, conf):
        """
        Configure plugin.
        """
        super().configure(options, conf)
        if options.warnaserror:
            self.enabled = True


    def prepareTestRunner(self, runner):
        """
        Treat warnings as errors.
        """
        if self.enabled:
            return WarnAsErrorTestRunner(runner)
        else:
            return runner


class WarnAsErrorTestRunner(object):


    def __init__(self, runner):
        self.runner=runner


    def run(self, test):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Filter out some deprecation warnings inside nose 1.3.7 when run
            # on python 3.5b2. See
            #     https://github.com/nose-devs/nose/issues/929
            warnings.filterwarnings(
                "ignore",
                message=".*getargspec.*",
                category=DeprecationWarning,
                module="nose|scipy"
            )
            # Filter out some deprecation warnings inside matplotlib on Python
            # 3.4
            warnings.filterwarnings(
                "ignore",
                message=".*elementwise.*",
                category=DeprecationWarning,
                module="matplotlib"
            )

            return self.runner.run(test)
