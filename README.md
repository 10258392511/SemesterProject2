# Repository for Semester Project 2
To install the project as an editable package, please use at the root directory:
```bash
pip install -e .
```
### Preliminary experiments:
- **DDQN**
  - CartPole-v0 (Before & After Training):
    <table>
      <tr>
         <td><img src="results/dqn_cart_pole/dqn_cart_pole_episode_1.gif" width=300 height=300></td>
         <td><img src="results/dqn_cart_pole/dqn_cart_pole_episode_300.gif" width=300 height=300></td>
      </tr>
    </table>
    
    
  - LunarLander-v3 (Before & After Training):
    <table>
      <tr>
         <td><img src="results/dqn_lunar_lander/dqn_lunar_lander_episode_1.gif" width=300 height=300></td>
         <td><img src="results/dqn_lunar_lander/dqn_lunar_lander_episode_500.gif" width=300 height=300></td>
      </tr>
    </table>

### Random Sampling Policy:
<table>
  <tr>
    <td><img src="results/volumetric_sampling_policy/no_title.gif" width=330 height=480></td>
    <td><img src="results/volumetric_sampling_policy/title.gif" width=750 height=480></td>
  </tr>
 </table>

### Greedy Policy
Without Inhibition of Return (IoR); with IoR, episodes 25 & 30
<table>
  <tr>
    <td><img src="results/vit_greedy/train_without_inhibition_of_return_ep_15.gif" width=200 height=300></td>
    <td><img src="results/vit_greedy/train_with_inhibition_of_return_ep_25.gif" width=200 height=300></td>
    <td><img src="results/vit_greedy/train_with_inhibition_of_return_ep_30.gif" width=200 height=300></td>
   <tr>
</table>


### Grid of Agents in a Testing Volume
<table>

[comment]: <> (  <tr>)

[comment]: <> (    <td><img src="results/vit_greedy/test_1/agent_0.gif" width=200 height=300></td>)

[comment]: <> (    <td><img src="results/vit_greedy/test_1/agent_1.gif" width=200 height=300></td>)

[comment]: <> (    <td><img src="results/vit_greedy/test_1/agent_2.gif" width=200 height=300></td>)

[comment]: <> (    <td><img src="results/vit_greedy/test_1/agent_3.gif" width=200 height=300></td>)

[comment]: <> (  <tr>)
  
[comment]: <> (  <tr>)

[comment]: <> (    <td><img src="results/vit_greedy/test_1/agent_4.gif" width=200 height=300></td>)

[comment]: <> (    <td><img src="results/vit_greedy/test_1/agent_5.gif" width=200 height=300></td>)

[comment]: <> (    <td><img src="results/vit_greedy/test_1/agent_6.gif" width=200 height=300></td>)

[comment]: <> (    <td><img src="results/vit_greedy/test_1/agent_7.gif" width=200 height=300></td>)

[comment]: <> (  <tr>)


   <tr> 
        <td><img src="results/vit_greedy/test_16/agent_0.gif" width="200" height=300"></td>
        <td><img src="results/vit_greedy/test_16/agent_1.gif" width="200" height=300"></td>
    </tr>
</table>
