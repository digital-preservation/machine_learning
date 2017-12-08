#! /usr/bin/env ruby
require 'spec_helper'
require 'puppet/face'

module PuppetFaceSpecs 
describe Puppet::Face[:config, '0.0.1'] do

  FS = Puppet::FileSystem

  it "prints a single setting without the name" do
    Puppet[:trace] = true

    expect { subject.print("trace") }.to have_printed('true')
  end

  it "prints multiple settings with the names" do
    Puppet[:trace] = true
    Puppet[:syslogfacility] = "file"

    expect { subject.print("trace", "syslogfacility") }.to have_printed(<<-OUTPUT)
syslogfacility = file
trace = true
    OUTPUT
  end

  it "prints the setting from the selected section" do
    Puppet.settings.parse_config(<<-CONF)
    [user]
    syslogfacility = file
    CONF

    expect { subject.print("syslogfacility", :section => "user") }.to have_printed('file')
  end

  it "defaults to all when no arguments are given" do
    subject.expects(:puts).times(Puppet.settings.to_a.length)

    subject.print
  end

  it "prints out all of the settings when asked for 'all'" do
    subject.expects(:puts).times(Puppet.settings.to_a.length)

    subject.print('all')
  end

  context "when setting config values" do
    let(:config_file) { '/foo/puppet.conf' }
    let(:path) { Pathname.new(config_file).expand_path }
    before(:each) do
      Puppet[:config] = config_file
      Puppet::FileSystem.stubs(:pathname).with(path.to_s).returns(path)
      Puppet::FileSystem.stubs(:touch)
    end

    it "writes to the correct puppet config file" do
      Puppet::FileSystem.expects(:open).with(path, anything, anything)
      subject.set('foo', 'bar')
    end

    it "creates a config file if one does not exist" do
      Puppet::FileSystem.stubs(:open).with(path, anything, anything).yields(StringIO.new)
      Puppet::FileSystem.expects(:touch).with(path)
      subject.set('foo', 'bar')
    end

    it "sets the supplied config/value in the default section (main)" do
      Puppet::FileSystem.stubs(:open).with(path, anything, anything).yields(StringIO.new)
      config = Puppet::Settings::IniFile.new([Puppet::Settings::IniFile::DefaultSection.new])
      manipulator = Puppet::Settings::IniFile::Manipulator.new(config)
      Puppet::Settings::IniFile::Manipulator.stubs(:new).returns(manipulator)

      manipulator.expects(:set).with("main", "foo", "bar")
      subject.set('foo', 'bar')
    end

    it "sets the value in the supplied section" do
      Puppet::FileSystem.stubs(:open).with(path, anything, anything).yields(StringIO.new)
      config = Puppet::Settings::IniFile.new([Puppet::Settings::IniFile::DefaultSection.new])
      manipulator = Puppet::Settings::IniFile::Manipulator.new(config)
      Puppet::Settings::IniFile::Manipulator.stubs(:new).returns(manipulator)

      manipulator.expects(:set).with("baz", "foo", "bar")
      subject.set('foo', 'bar', {:section => "baz"})

    end

    it "opens the file with UTF-8 encoding" do
      Puppet::FileSystem.expects(:open).with(path, nil, 'r+:UTF-8')
      subject.set('foo', 'bar')
    end
  end

  shared_examples_for :config_printing_a_section do |section|

    def add_section_option(args, section)
      args << { :section => section } if section
      args
    end

    it "prints directory env settings for an env that exists" do
      FS.overlay(
        FS::MemoryFile.a_directory(File.expand_path("/dev/null/environments"), [
          FS::MemoryFile.a_directory("production", [
            FS::MemoryFile.a_missing_file("environment.conf"),
          ]),
        ])
      ) do
        args = "environmentpath","manifest","modulepath","environment","basemodulepath"
        expect { subject.print(*add_section_option(args, section)) }.to have_printed(<<-OUTPUT)
basemodulepath = #{File.expand_path("/some/base")}
environment = production
environmentpath = #{File.expand_path("/dev/null/environments")}
manifest = #{File.expand_path("/dev/null/environments/production/manifests")}
modulepath = #{File.expand_path("/dev/null/environments/production/modules")}#{File::PATH_SEPARATOR}#{File.expand_path("/some/base")}
        OUTPUT
      end
    end

    it "interpolates settings in environment.conf" do
      FS.overlay(
        FS::MemoryFile.a_directory(File.expand_path("/dev/null/environments"), [
          FS::MemoryFile.a_directory("production", [
            FS::MemoryFile.a_regular_file_containing("environment.conf", <<-CONTENT),
            modulepath=/custom/modules#{File::PATH_SEPARATOR}$basemodulepath
            CONTENT
          ]),
        ])
      ) do
        args = "environmentpath","manifest","modulepath","environment","basemodulepath"
        expect { subject.print(*add_section_option(args, section)) }.to have_printed(<<-OUTPUT)
basemodulepath = #{File.expand_path("/some/base")}
environment = production
environmentpath = #{File.expand_path("/dev/null/environments")}
manifest = #{File.expand_path("/dev/null/environments/production/manifests")}
modulepath = #{File.expand_path("/custom/modules")}#{File::PATH_SEPARATOR}#{File.expand_path("/some/base")}
        OUTPUT
      end
    end

    it "prints the default configured env settings for an env that does not exist" do
      Puppet[:environment] = 'doesnotexist'

      FS.overlay(
        FS::MemoryFile.a_directory(File.expand_path("/dev/null/environments"), [
          FS::MemoryFile.a_missing_file("doesnotexist")
        ])
      ) do
        args = "environmentpath","manifest","modulepath","environment","basemodulepath"
        expect { subject.print(*add_section_option(args, section)) }.to have_printed(<<-OUTPUT)
basemodulepath = #{File.expand_path("/some/base")}
environment = doesnotexist
environmentpath = #{File.expand_path("/dev/null/environments")}
manifest = 
modulepath = 
        OUTPUT
      end
    end
  end

  context "when printing environment settings" do
    context "from main section" do
      before(:each) do
        Puppet.settings.parse_config(<<-CONF)
        [main]
        environmentpath=$confdir/environments
        basemodulepath=/some/base
        CONF
      end

      it_behaves_like :config_printing_a_section, nil
    end

    context "from master section" do

      before(:each) do
        Puppet.settings.parse_config(<<-CONF)
        [master]
        environmentpath=$confdir/environments
        basemodulepath=/some/base
        CONF
      end

      it_behaves_like :config_printing_a_section, :master
    end
  end
end
end
