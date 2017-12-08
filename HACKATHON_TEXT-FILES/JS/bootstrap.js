/*
  This file is part of the PhantomJS project from Ofi Labs.

  Copyright (C) 2011 Ariya Hidayat <ariya.hidayat@gmail.com>
  Copyright (C) 2011 Ivan De Marino <ivan.de.marino@gmail.com>
  Copyright (C) 2011 James Roe <roejames12@hotmail.com>
  Copyright (C) 2011 execjosh, http://execjosh.blogspot.com
  Copyright (C) 2012 James M. Greene <james.m.greene@gmail.com>

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

phantom.__defineErrorSignalHandler__ = function (obj, page, handlers) {
  var handlerName = 'onError';

  Object.defineProperty(obj, handlerName, {
    set: function (f) {
      // Disconnect previous handler (if any)
      var handlerObj = handlers[handlerName];
      if (!!handlerObj && typeof handlerObj.callback === 'function' && typeof handlerObj.connector === 'function') {
        try {
          page.javaScriptErrorSent.disconnect(handlerObj.connector);
        } catch (e) {}
      }

      // Delete the previous handler
      delete handlers[handlerName];

      if (typeof f === 'function') {
        var connector = function (message, lineNumber, source, stack) {
          var revisedStack = JSON.parse(stack).map(function (item) {
            return {
              file: item.url,
              line: item.lineNumber,
              function: item.functionName
            };
          });
          if (revisedStack.length == 0)
            revisedStack = [{
              file: source,
              line: lineNumber
            }];

          f(message, revisedStack);
        };
        // Store the new handler for reference
        handlers[handlerName] = {
          callback: f,
          connector: connector
        };

        page.javaScriptErrorSent.connect(connector);
      }
    },
    get: function () {
      var handlerObj = handlers[handlerName];
      return (!!handlerObj && typeof handlerObj.callback === 'function' && typeof handlerObj.connector === 'function') ?
        handlers[handlerName].callback :
        undefined;
    },
    configurable: true
  });
};

(function () {
  var handlers = {};
  phantom.__defineErrorSignalHandler__(phantom, phantom.page, handlers);
})();

// TODO: Make this output to STDERR
phantom.defaultErrorHandler = function (message, stack) {
  console.log(message + '\n');

  stack.forEach(function (item) {
    var message = item.file + ':' + item.line;
    if (item['function'])
      message += ' in ' + item['function'];
    console.log('  ' + message);
  });
};

phantom.onError = phantom.defaultErrorHandler;

phantom.callback = function (callback) {
  var ret = phantom.createCallback();
  ret.called.connect(function (args) {
    var retVal = callback.apply(this, args);
    ret.returnValue = retVal;
  });
  return ret;
};

(function () {
  // CommonJS module implementation follows

  window.global = window;
  // fs is loaded at the end, when everything is ready
  var fs;
  var cache = {};
  var paths = [];
  // use getters to initialize lazily
  // (for future, now both fs and system are loaded anyway)
  var nativeExports = {
    get fs() {
      return phantom.createFilesystem();
    },
    get child_process() {
      return phantom._createChildProcess();
    },
    get system() {
      return phantom.createSystem();
    }
  };
  var extensions = {
    '.js': function (module, filename) {
      var code = fs.read(filename);
      module._compile(code);
    },

    '.json': function (module, filename) {
      module.exports = JSON.parse(fs.read(filename));
    }
  };

  function loadFs() {
    var file, code, module, filename = ':/modules/fs.js';

    module = new Module(filename);
    cache[filename] = module;
    module.exports = nativeExports.fs;

    file = module.exports._open(filename, {
      mode: 'r'
    });
    code = file.read();
    file.close();
    module._compile(code);

    return module.exports;
  }

  function dirname(path) {
    var replaced = path.replace(/\/[^/]*\/?$/, '');
    if (replaced == path) {
      replaced = '';
    }
    return replaced;
  }

  function basename(path) {
    return path.replace(/.*\//, '');
  }

  function joinPath() {
    // It should be okay to hard-code a slash here.
    // The FileSystem module returns a platform-specific
    // separator, but the JavaScript engine only expects
    // the slash.
    var args = Array.prototype.slice.call(arguments);
    return args.join('/');
  }

  function tryFile(path) {
    if (fs.isFile(path)) return path;
    return null;
  }

  function tryExtensions(path) {
    var filename, exts = Object.keys(extensions);
    for (var i = 0; i < exts.length; ++i) {
      filename = tryFile(path + exts[i]);
      if (filename) return filename;
    }
    return null;
  }

  function tryPackage(path) {
    var filename, package, packageFile = joinPath(path, 'package.json');
    if (fs.isFile(packageFile)) {
      package = JSON.parse(fs.read(packageFile));
      if (!package || !package.main) return null;

      filename = fs.absolute(joinPath(path, package.main));

      return tryFile(filename) || tryExtensions(filename) ||
        tryExtensions(joinPath(filename, 'index'));
    }
    return null;
  }

  function Module(filename, stubs) {
    if (filename) this._setFilename(filename);
    this.exports = {};
    this.stubs = {};
    for (var name in stubs) {
      this.stubs[name] = stubs[name];
    }
  }

  Module.prototype._setFilename = function (filename) {
    this.id = this.filename = filename;
    this.dirname = dirname(filename);
  };

  Module.prototype._isNative = function () {
    return this.filename && this.filename[0] === ':';
  };

  Module.prototype._getPaths = function (request) {
    var _paths = [],
      dir;

    if (request[0] === '.') {
      _paths.push(fs.absolute(joinPath(phantom.webdriverMode ? ':/ghostdriver' : this.dirname, request)));
    } else if (fs.isAbsolute(request)) {
      _paths.push(fs.absolute(request));
    } else {
      // first look in PhantomJS modules
      _paths.push(joinPath(':/modules', request));
      // then look in node_modules directories
      if (!this._isNative()) {
        dir = this.dirname;
        while (dir) {
          _paths.push(joinPath(dir, 'node_modules', request));
          dir = dirname(dir);
        }
      }
    }

    for (var i = 0; i < paths.length; ++i) {
      if (fs.isAbsolute(paths[i])) {
        _paths.push(fs.absolute(joinPath(paths[i], request)));
      } else {
        _paths.push(fs.absolute(joinPath(this.dirname, paths[i], request)));
      }
    }

    return _paths;
  };

  Module.prototype._getFilename = function (request) {
    var path, filename = null,
      _paths = this._getPaths(request);

    for (var i = 0; i < _paths.length && !filename; ++i) {
      path = _paths[i];
      filename = tryFile(path) || tryExtensions(path) || tryPackage(path) ||
        tryExtensions(joinPath(path, 'index'));
    }

    return filename;
  };

  Module.prototype._getRequire = function () {
    var self = this;

    function require(request) {
      return self.require(request);
    }
    require.cache = cache;
    require.extensions = extensions;
    require.paths = paths;
    require.stub = function (request, exports) {
      self.stubs[request] = {
        exports: exports
      };
    };

    return require;
  };

  Module.prototype._load = function () {
    var ext = this.filename.match(/\.[^.]+$/)[0];
    if (!ext) ext = '.js';
    extensions[ext](this, this.filename);
  };

  Module.prototype._compile = function (code) {
    phantom.loadModule(code, this.filename);
  };

  Module.prototype.require = function (request) {
    var filename, module;

    // first see if there are any stubs for the request
    if (this.stubs.hasOwnProperty(request)) {
      if (this.stubs[request].exports instanceof Function) {
        this.stubs[request].exports = this.stubs[request].exports();
      }
      return this.stubs[request].exports;
    }

    // else look for a file
    filename = this._getFilename(request);
    if (!filename) {
      throw new Error(`Cannot find module "${request}"`);
    }

    if (cache.hasOwnProperty(filename)) {
      return cache[filename].exports;
    }

    module = new Module(filename, this.stubs);
    if (module._isNative()) {
      module.exports = nativeExports[request] || {};
    }
    cache[filename] = module;
    module._load();

    return module.exports;
  };

  (function () {
    var cwd, mainFilename, mainModule = new Module();
    window.require = mainModule._getRequire();
    fs = loadFs();
    cwd = fs.absolute(phantom.libraryPath);
    mainFilename = joinPath(cwd, basename(require('system').args[0]) || 'repl');
    mainModule._setFilename(mainFilename);
  }());
}());

// Legacy way to use WebPage
window.WebPage = require('webpage').create;
