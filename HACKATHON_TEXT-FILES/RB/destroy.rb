test_name "should delete an email alias"

confine :except, :platform => 'windows'

tag 'audit:low',
    'audit:refactor',  # Use block style `test_name`
    'audit:acceptance' # Could be done at the integration (or unit) layer though
                       # actual changing of resources could irreparably damage a
                       # host running this, or require special permissions.

name = "pl#{rand(999999).to_i}"
agents.each do |agent|
  teardown do
    #(teardown) restore the alias file
    on(agent, "mv /tmp/aliases /etc/aliases", :acceptable_exit_codes => [0,1])
  end

  #------- SETUP -------#
  step "(setup) backup alias file"
  on(agent, "cp /etc/aliases /tmp/aliases", :acceptable_exit_codes => [0,1])

  step "(setup) create a mailalias"
  on(agent, "echo '#{name}: foo,bar,baz' >> /etc/aliases")

  step "(setup) verify the alias exists"
  on(agent, "cat /etc/aliases")  do |res|
    assert_match(/#{name}:.*foo,bar,baz/, res.stdout, "mailalias not in aliases file")
  end

  #------- TESTS -------#
  step "delete the aliases database with puppet"
  args = ['ensure=absent',
          'recipient="foo,bar,baz"']
  on(agent, puppet_resource('mailalias', name, args))


  step "verify the alias is absent"
  on(agent, "cat /etc/aliases")  do |res|
    assert_no_match(/#{name}:.*foo,bar,baz/, res.stdout, "mailalias was not removed from aliases file")
  end
end  
