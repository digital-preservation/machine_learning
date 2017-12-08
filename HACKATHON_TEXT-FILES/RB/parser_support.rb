require 'puppet/parser/functions'
require 'puppet/parser/files'
require 'puppet/resource/type_collection'
require 'puppet/resource/type'
require 'monitor'

module Puppet::Pops
module Parser
# Supporting logic for the parser.
# This supporting logic has slightly different responsibilities compared to the original Puppet::Parser::Parser.
# It is only concerned with parsing.
#
class Parser
  # Note that the name of the contained class and the file name (currently parser_support.rb)
  # needs to be different as the class is generated by Racc, and this file (parser_support.rb) is included as a mix in
  #

  # Simplify access to the Model factory
  # Note that the parser/parser support does not have direct knowledge about the Model.
  # All model construction/manipulation is made by the Factory.
  #
  Factory = Model::Factory

  attr_accessor :lexer
  attr_reader :definitions

  # Returns the token text of the given lexer token, or nil, if token is nil
  def token_text t
    return t if t.nil?
    if t.is_a?(Factory) && t.model_class <= Model::QualifiedName
      t['value']
    elsif t.is_a?(Model::QualifiedName)
      t.value
    else
      # else it is a lexer token
      t[:value]
    end
  end

  # Produces the fully qualified name, with the full (current) namespace for a given name.
  #
  # This is needed because class bodies are lazily evaluated and an inner class' container(s) may not
  # have been evaluated before some external reference is made to the inner class; its must therefore know its complete name
  # before evaluation-time.
  #
  def classname(name)
    [namespace, name].join('::').sub(/^::/, '')
  end

  # Raises a Parse error with location information. Information about file is always obtained from the
  # lexer. Line and position is produced if the given semantic is a Positioned object and have been given an offset.
  #
  def error(semantic, message)
    except = Puppet::ParseError.new(message)
    if semantic.is_a?(LexerSupport::TokenValue)
      except.file = semantic[:file];
      except.line = semantic[:line];
      except.pos = semantic[:pos];
    else
      locator = @lexer.locator
      except.file = locator.file
      if semantic.is_a?(Factory)
        offset = semantic['offset']
        unless offset.nil?
          except.line = locator.line_for_offset(offset)
          except.pos = locator.pos_on_line(offset)
        end
      end
    end
    raise except
  end

  # Parses a file expected to contain pp DSL logic.
  def parse_file(file)
    unless Puppet::FileSystem.exist?(file)
      unless file =~ /\.pp$/
        file = file + ".pp"
      end
    end
    @lexer.file = file
    _parse
  end

  def initialize()
    @lexer = Lexer2.new
    @namestack = []
    @definitions = []
  end

  # This is a callback from the generated parser (when an error occurs while parsing)
  #
  def on_error(token,value,stack)
    if token == 0 # denotes end of file
      value_at = 'end of input'
    else
      value_at = "'#{value[:value]}'"
    end
    error = Issues::SYNTAX_ERROR.format(:where => value_at)
    error = "#{error}, token: #{token}" if @yydebug

    # Note, old parser had processing of "expected token here" - do not try to reinstate:
    # The 'expected' is only of value at end of input, otherwise any parse error involving a
    # start of a pair will be reported as expecting the close of the pair - e.g. "$x.each |$x {|", would
    # report that "seeing the '{', the '}' is expected. That would be wrong.
    # Real "expected" tokens are very difficult to compute (would require parsing of racc output data). Output of the stack
    # could help, but can require extensive backtracking and produce many options.
    #
    # The lexer should handle the "expected instead of end of file for strings, and interpolation", other expectancies
    # must be handled by the grammar. The lexer may have enqueued tokens far ahead - the lexer's opinion about this
    # is not trustworthy.
    #
    file = nil
    line = nil
    pos  = nil
    if token != 0
      file = value[:file]
      line = value[:line]
      pos  = value[:pos]
    else
      # At end of input, use what the lexer thinks is the source file
      file = lexer.file
    end
    file = nil unless file.is_a?(String) && !file.empty?
    raise Puppet::ParseErrorWithIssue.new(error, file, line, pos, nil, issue_code = Issues::SYNTAX_ERROR.issue_code)
  end

  # Parses a String of pp DSL code.
  #
  def parse_string(code, path = nil)
    @lexer.lex_string(code, path)
    _parse()
  end

  # Mark the factory wrapped model object with location information
  # @return [Factory] the given factory
  # @api private
  #
  def loc(factory, start_locatable, end_locatable = nil)
    factory.record_position(@lexer.locator, start_locatable, end_locatable)
  end

  # Mark the factory wrapped heredoc model object with location information
  # @return [Factory] the given factory
  # @api private
  #
  def heredoc_loc(factory, start_locatable, end_locatable = nil)
    factory.record_heredoc_position(start_locatable, end_locatable)
  end

  def aryfy(o)
    o = [o] unless o.is_a?(Array)
    o
  end

  def namespace
    @namestack.join('::')
  end

  def namestack(name)
    @namestack << name
  end

  def namepop()
    @namestack.pop
  end

  def add_definition(definition)
    @definitions << definition.model
    definition
  end

  def add_mapping(produces)
    # The actual handling of mappings happens in PopsBridge
    add_definition(produces)
  end

  # Transforms an array of expressions containing literal name expressions to calls if followed by an
  # expression, or expression list
  #
  def transform_calls(expressions)
    # Factory transform raises an error if a non qualified name is followed by an argument list
    # since there is no way that that can be transformed back to sanity. This occurs in situations like this:
    #
    #  $a = 10, notice hello
    #
    # where the "10, notice" forms an argument list. The parser builds an Array with the expressions and includes
    # the comma tokens to enable the error to be reported against the first comma.
    #
    begin
      Factory.transform_calls(expressions)
    rescue Factory::ArgsToNonCallError => e
      # e.args[1] is the first comma token in the list
      # e.name_expr is the function name expression
      if e.name_expr.is_a?(Factory) && e.name_expr.model_class <= Model::QualifiedName
        error(e.args[1], _("attempt to pass argument list to the function '%{name}' which cannot be called without parentheses") % { name: e.name_expr['value'] })
      else
        error(e.args[1], _("illegal comma separated argument list"))
      end
    end
  end

  # Transforms a LEFT followed by the result of attribute_operations, this may be a call or an invalid sequence
  def transform_resource_wo_title(left, resource, lbrace_token, rbrace_token)
    Factory.transform_resource_wo_title(left, resource, lbrace_token, rbrace_token)
  end

  # Creates a program with the given body.
  #
  def create_program(body)
    locator = @lexer.locator
    Factory.PROGRAM(body, definitions, locator)
  end

  # Creates an empty program with a single No-op at the input's EOF offset with 0 length.
  #
  def create_empty_program()
    locator = @lexer.locator
    no_op = Factory.literal(nil)
    # Create a synthetic NOOP token at EOF offset with 0 size. The lexer does not produce an EOF token that is
    # visible to the grammar rules. Creating this token is mainly to reuse the positioning logic as it
    # expects a token decorated with location information.
    token_sym, token = @lexer.emit_completed([:NOOP,'',0], locator.string.bytesize)
    loc(no_op, token)
    # Program with a Noop
    program = Factory.PROGRAM(no_op, [], locator)
    program
  end

  # Performs the parsing and returns the resulting model.
  # The lexer holds state, and this is setup with {#parse_string}, or {#parse_file}.
  #
  # @api private
  #
  def _parse()
    begin
      @yydebug = false
      main = yyparse(@lexer,:scan)
    end
    return main
  ensure
    @lexer.clear
    @namestack = []
    @definitions = []
  end
end
end
end
