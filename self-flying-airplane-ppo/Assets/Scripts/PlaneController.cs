using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using Random=UnityEngine.Random;
//using UnityEngine.SceneManagement;
using UnityEngine.UI;

using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;


public class PlaneController : Agent
{   
    // Begin - Controller Variables
    public float forwardSpeed = 45.0f;
    public float gravityMultiplier = 20.0f; //Higher value will increase or decrease speed faster depending on going up or down.
    public float turnSpeedMultiplier = 2.0f; // Higher value will make the airplane rotate faster.
    public float maxSpeed = 100.0f;
    // End - Controller Variables
   
    // Variables for ending epsiode
    public int stepTimeOut = 1200;
    private float nextStepTimeout;
    public float currentStepCount;
    private bool newEpisode;
    private bool agentCrashed;

    //Checkpoints
    public int currentCheckpointIndex = 0;
    public GameObject[] checkpointsArray;

    //Overlay text
    private Text reloadCount;
    public GameObject txt;
    static private int count;

    //Rewards
    public float prevReward;
    public float currentReward;

    //For observations
    private Vector3 targetPos;
    public float distanceToTarget;
    public float normDistToTarget;
    public float initialDistanceToTarget;
    

    //Variables for the Runway and Playfield
    public Transform Runway, ArenaBoundary;
    private float runwayX, runwayZ, arenaX, arenaZ;
    private Vector3 runwayCenter, arenaCenter;

    //Actions Movements
    private int directionX, directionZ;


    // Start is called before the first frame update
    void Start()
    {
        targetPos = checkpointsArray[currentCheckpointIndex].transform.localPosition;
        initialDistanceToTarget = Vector3.Distance(this.transform.localPosition, targetPos);
        Debug.Log("Target position found at: " + targetPos);
        currentReward = 0.0f;
        prevReward = 0.0f;
        newEpisode = true;

        // Init Text overlay component.
        reloadCount = txt.GetComponent<Text>();

        // Init Runway and ArenaBoundary planes.
        InitPlanes();

    }

    void InitPlanes(){
        // Init Runway and ArenaBoundary planes.
        runwayX = Runway.GetComponent<Renderer>().bounds.size.x / 2;
        runwayZ = Runway.GetComponent<Renderer>().bounds.size.z / 2;
        runwayCenter = Runway.GetComponent<Renderer>().bounds.center;

        arenaX = ArenaBoundary.GetComponent<Renderer>().bounds.size.x / 2;
        arenaZ = ArenaBoundary.GetComponent<Renderer>().bounds.size.z / 2;
        arenaCenter = ArenaBoundary.GetComponent<Renderer>().bounds.center;
    }

    void Update(){
        currentStepCount = nextStepTimeout - this.StepCount;
        reloadCount.text = "Count: " + count.ToString() + "\n" + "Prev Reward: " + prevReward.ToString();
    }

    
    private void OnTriggerEnter(Collider col){
            if(newEpisode && col.tag == "Terrain"){
                agentCrashed = true;
            }    
    }


    


    // Set up an Agent instance at the beginning of an episode
    public override void OnEpisodeBegin() {
        
        // Update Text Overlay
        count = count + 1;
        // Spawn Agent
        SpawnAgent();
        // Reset next checkpoint
        ResetCheckpoints();
        // Update the step timeout
     

        currentReward = GetCumulativeReward();
    }

    //Spawns agent at random location within the Runway plane.
    public void SpawnAgent(){
        var agentCoordsX = runwayCenter.x + Random.Range(-runwayX, runwayX)*0.001f;
        var agentCoordsZ = runwayCenter.z + Random.Range(-runwayZ, runwayZ)*0.001f;
        Vector3 agentRot = new Vector3(0.0f, Random.Range(-180, -170), 0.0f);

        this.transform.position = new Vector3(agentCoordsX, 90.0f, agentCoordsZ);
        this.transform.rotation = Quaternion.Euler(agentRot);

        nextStepTimeout = this.StepCount + stepTimeOut;
        newEpisode = true;
        agentCrashed = false;
    }

    public void ResetCheckpoints(){
        currentCheckpointIndex = 0;
        targetPos = checkpointsArray[currentCheckpointIndex].transform.localPosition;
        initialDistanceToTarget = Vector3.Distance(this.transform.localPosition, targetPos);
        distanceToTarget = Vector3.Distance(this.transform.localPosition, targetPos);
    }


    /* Collect the vector observation of the agent for the current step. 
       Ray Perception Sensors are added automaticly by the mlagent package
       and should not be added manually below.  */
   public override void CollectObservations(VectorSensor sensor)
    {   
        distanceToTarget = Vector3.Distance(this.transform.localPosition, targetPos); 
        normDistToTarget = distanceToTarget/initialDistanceToTarget;
  
        //Normalized observation values
        sensor.AddObservation(normDistToTarget);

    }

    /* Giving the agent actions corresponding to the same controllers WASD
       on the keyboard. What happens when either action is executed are handled 
       in the Update() function, i.e. how the airplane moves.
       NOTE: Must set Behaviour type to: Heuristics Only for the agent to use this*/
   public override void Heuristic(in ActionBuffers actionsOut)
    {   
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = 2f * Mathf.Clamp(Input.GetAxis("Vertical"), -1f, 1f);
        continuousActionsOut[1] = 2f * Mathf.Clamp(Input.GetAxis("Horizontal"), -1f, 1f);
    }
    
    public void RewardAndEnd(float reward){
        AddReward(reward); 
        currentReward = GetCumulativeReward();
        prevReward = GetCumulativeReward();
        EndEpisode();
    }

    /* CURRENTLY NOT CALLED: Advanced Movements implemented but did not train properly. This 
       inlclude barrel rolls, and overall more realistic flight movements.*/
    void ProcessAdvancedMovement(float movementZ, float movementX){
        //Temporary solution, fix below code

        if (movementX < -0.5f) { directionX = -1; }
        if (movementX > 0.5f) { directionX = 1; }
        if (Math.Abs(movementX)< 0.5f) { directionX = 0; }
        if (movementZ < -0.5f) { directionZ = -1; }
        if (movementZ > 0.5f) { directionZ = 1; }
        if (Math.Abs(movementZ) < 0.5f) { directionZ = 0; }
        
        transform.position += transform.forward * Time.deltaTime * forwardSpeed;
        forwardSpeed -= transform.forward.y * Time.deltaTime * gravityMultiplier; //Simul
        transform.Rotate(directionX*turnSpeedMultiplier, 0.0f, -directionZ*turnSpeedMultiplier);
        //transform.Rotate(0.0f, directionZ*turnSpeedMultiplier, 0.0f);

        // Defines min and max speed of the agent.
        if (forwardSpeed < 50.0f){
            forwardSpeed = 50.0f;
        } 
        if (forwardSpeed > maxSpeed){
            forwardSpeed = maxSpeed;
        }

    }
    void ProcessMovement(float movementZ, float movementX){
    transform.position += transform.forward * Time.deltaTime * forwardSpeed;
    transform.Translate(transform.up * movementX*gravityMultiplier * Time.deltaTime); 
    forwardSpeed -= transform.forward.y * Time.deltaTime * gravityMultiplier; 
    transform.Rotate(0.0f, movementZ*turnSpeedMultiplier, 0.0f);

}

    

    /* Below is used to specify the agent behavior at every step based on 
       the prodived action. */
    public override void OnActionReceived(ActionBuffers actionBuffers){
        //--------------------------ACTION----------------------------//
        var actionX = 2f * Mathf.Clamp(actionBuffers.ContinuousActions[0], -1f, 1f);
        var actionZ = 2f * Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f);

        //ProcessAdvancedMovement(actionZ, actionX);
        ProcessMovement(actionZ, actionX);
        //--------------------------REWARD----------------------------/  

        // Small negative reward every step
        AddReward(-1f/this.MaxStep);
        
       
        if (distanceToTarget < 30f) {
            CheckpointReached();
        }
        else if (newEpisode == true){
           
            if(this.StepCount > nextStepTimeout) {
                newEpisode = false;
                RewardAndEnd(-.5f);
               
            }
            else if(agentCrashed){
                newEpisode = false;
                agentCrashed = false;
                RewardAndEnd(-.5f);
            }
            else if (distanceToTarget > initialDistanceToTarget){
                newEpisode = false;
                RewardAndEnd(-.5f);
             
            }
            else if(AgentOutOfBounds()){
                newEpisode = false;
                RewardAndEnd(-.5f);
                
            }
        } 

        currentReward = GetCumulativeReward();
    }

    

    private bool AgentOutOfBounds(){
        if (this.transform.localPosition.y < 0 || this.transform.localPosition.y > 250 ||
            this.transform.localPosition.x > arenaCenter.x + arenaX || this.transform.localPosition.x < arenaCenter.x - arenaX ||
            this.transform.localPosition.z > arenaCenter.z + arenaZ || this.transform.localPosition.z > arenaCenter.z + arenaZ)
        {   
            return true;
        } else {
            return false;
        }
    }

    private void CheckpointReached(){
        
        if (currentCheckpointIndex == checkpointsArray.Length - 1){
            Debug.Log("LAST CHECKPOINT REACHED!"); 
            RewardAndEnd(1.0f);
        } else {
            AddReward(1.0f);
            Debug.Log("CHECKPOINT REACHED #" + currentCheckpointIndex);
            currentCheckpointIndex = currentCheckpointIndex + 1;
            nextStepTimeout = this.StepCount + stepTimeOut;
            targetPos = checkpointsArray[currentCheckpointIndex].transform.localPosition;
            initialDistanceToTarget = Vector3.Distance(this.transform.localPosition, targetPos);
            distanceToTarget = Vector3.Distance(this.transform.localPosition, targetPos);


        }
    }
}
