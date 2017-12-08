test_name "Package:IPS test for updatable (update, latest)"
confine :to, :platform => 'solaris-11'

tag 'audit:medium',
    'audit:refactor',  # Use block style `test_name`
    'audit:acceptance' # Could be done at the integration (or unit) layer though
                       # actual changing of resources could irreparably damage a
                       # host running this, or require special permissions.

require 'puppet/acceptance/solaris_util'
extend Puppet::Acceptance::IPSUtils

teardown do
  step "cleanup"
  agents.each do |agent|
    clean agent
  end
end


agents.each do |agent|
  step "IPS: setup"
  setup agent
  setup_fakeroot agent
  send_pkg agent, :pkg => 'mypkg@0.0.1'
  set_publisher agent

  step "IPS: basic - it should create"
  apply_manifest_on(agent, 'package {mypkg : ensure=>present}') do
    assert_match( /ensure: created/, result.stdout, "err: #{agent}")
  end

  step "IPS: ask to be latest"
  send_pkg agent, :pkg => 'mypkg@0.0.2'
  apply_manifest_on(agent, 'package {mypkg : ensure=>latest}')

  step "IPS: ensure it was upgraded"
  on agent, "pkg list -v mypkg" do
    assert_match( /mypkg@0.0.2/, result.stdout, "err: #{agent}")
  end

  step "IPS: when there are more than one option, choose latest."
  send_pkg agent,:pkg => 'mypkg@0.0.3'
  send_pkg agent,:pkg => 'mypkg@0.0.4'
  apply_manifest_on(agent, 'package {mypkg : ensure=>latest}')
  on agent, "pkg list -v mypkg" do
    assert_match( /mypkg@0.0.4/, result.stdout, "err: #{agent}")
  end
end
